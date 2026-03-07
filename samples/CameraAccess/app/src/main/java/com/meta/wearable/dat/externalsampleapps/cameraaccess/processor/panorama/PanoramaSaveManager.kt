package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/**
 * A panorama that has been saved to disk.
 *
 * @param id            Unique string key (timestamp millis as string).
 * @param timestamp     Epoch millis when the panorama was saved.
 * @param nodeCount     Number of Florence/hierarchy regions detected.
 * @param filePath      Absolute path to the full-resolution JPEG.
 * @param thumbnailBitmap  Pre-loaded thumbnail bitmap; null when not yet loaded.
 */
data class SavedPanorama(
    val id: String,
    val timestamp: Long,
    val nodeCount: Int,
    val filePath: String,
    val thumbnailBitmap: Bitmap? = null,
)


/**
 * Manages persistence of stitched (and annotated) panorama bitmaps.
 *
 * Files are stored in `<externalFilesDir>/panoramas/` as JPEG pairs:
 *   - `panorama_<ts>_n<count>.jpg`       — full resolution
 *   - `panorama_<ts>_n<count>_thumb.jpg` — 200 px tall thumbnail
 */
class PanoramaSaveManager(context: Context) {

    companion object {
        private const val TAG = "PanoramaSaveManager"
        private const val THUMB_HEIGHT = 200
        private val NAMED_PATTERN = Regex("""panorama_(\d+)_n(\d+)\.jpg""")
    }

    private val dir = File(context.getExternalFilesDir(null), "panoramas")
        .also { it.mkdirs() }

    /**
     * Save [bitmap] to disk and return a [SavedPanorama] record.
     * Blocking — call on a background thread.
     */
    fun save(bitmap: Bitmap, nodeCount: Int): SavedPanorama {
        val ts   = System.currentTimeMillis()
        val stem = "panorama_${ts}_n${nodeCount}"

        // Full-resolution JPEG
        val fullFile = File(dir, "$stem.jpg")
        FileOutputStream(fullFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }

        // Thumbnail
        val thumbH = THUMB_HEIGHT
        val thumbW = (bitmap.width.toFloat() / bitmap.height * thumbH).toInt().coerceAtLeast(1)
        val thumb  = Bitmap.createScaledBitmap(bitmap, thumbW, thumbH, true)
        val thumbFile = File(dir, "${stem}_thumb.jpg")
        FileOutputStream(thumbFile).use { thumb.compress(Bitmap.CompressFormat.JPEG, 80, it) }
        thumb.recycle()

        Log.d(TAG, "Saved panorama: ${fullFile.name} (${fullFile.length() / 1024} KB)")
        return SavedPanorama(id = ts.toString(), timestamp = ts, nodeCount = nodeCount, filePath = fullFile.absolutePath)
    }

    /**
     * List all saved panoramas, newest-first, with thumbnails loaded.
     *
     * Accepts two kinds of files in the panoramas directory:
     *  - Files matching `panorama_<ts>_n<count>.jpg` — saved by [save]; timestamp and
     *    region count are decoded from the filename.
     *  - Any other `.jpg` file (dropped manually via USB/ADB) — timestamp comes from
     *    the file's last-modified time, region count defaults to 0, and a thumbnail is
     *    generated on first access.
     *
     * Blocking — call on a background thread.
     */
    fun listSaved(): List<SavedPanorama> {
        val allJpegs = dir.listFiles { f ->
            f.isFile && f.name.lowercase().endsWith(".jpg") && !f.name.endsWith("_thumb.jpg")
        } ?: return emptyList()

        return allJpegs.mapNotNull { file ->
            val named = NAMED_PATTERN.matchEntire(file.name)
            val ts = named?.groupValues?.get(1)?.toLongOrNull() ?: file.lastModified()
            val n  = named?.groupValues?.get(2)?.toIntOrNull()  ?: 0
            val id = ts.toString()

            // Locate or create thumbnail
            val thumbFile = File(dir, "${file.nameWithoutExtension}_thumb.jpg")
            if (!thumbFile.exists()) {
                val full = BitmapFactory.decodeFile(file.absolutePath) ?: return@mapNotNull null
                val thumbH = THUMB_HEIGHT
                val thumbW = (full.width.toFloat() / full.height * thumbH).toInt().coerceAtLeast(1)
                val thumb  = Bitmap.createScaledBitmap(full, thumbW, thumbH, true)
                try {
                    FileOutputStream(thumbFile).use { thumb.compress(Bitmap.CompressFormat.JPEG, 80, it) }
                } catch (e: Exception) {
                    Log.w(TAG, "Could not write thumbnail for ${file.name}: ${e.message}")
                }
                thumb.recycle()
                full.recycle()
            }

            val thumb = BitmapFactory.decodeFile(thumbFile.absolutePath)
            SavedPanorama(
                id              = id,
                timestamp       = ts,
                nodeCount       = n,
                filePath        = file.absolutePath,
                thumbnailBitmap = thumb,
            )
        }.sortedByDescending { it.timestamp }
    }

    /**
     * Delete both the full-resolution JPEG and its thumbnail for [id].
     * Blocking — call on a background thread.
     */
    fun delete(id: String) {
        // Find the main file whose derived timestamp matches [id], then delete it,
        // its thumbnail, and its .glassio sidecar if present.
        dir.listFiles { f ->
            f.isFile && f.name.lowercase().endsWith(".jpg") && !f.name.endsWith("_thumb.jpg")
        }?.forEach { file ->
            val named = NAMED_PATTERN.matchEntire(file.name)
            val ts = named?.groupValues?.get(1)?.toLongOrNull() ?: file.lastModified()
            if (ts.toString() == id) {
                val stem = file.nameWithoutExtension
                file.delete()
                File(dir, "${stem}_thumb.jpg").takeIf { it.exists() }?.delete()
                File(dir, "$stem.glassio").takeIf    { it.exists() }?.delete()
                Log.d(TAG, "Deleted panorama $id (${file.name})")
            }
        }
    }

    /**
     * Load the full-resolution bitmap for [id].
     * Works for both named files (`panorama_<ts>_n<count>.jpg`) and externally-dropped
     * files whose id is derived from [File.lastModified].
     * Blocking — call on a background thread.
     */
    fun load(id: String): Bitmap? {
        val file = findJpegFileForId(id) ?: return null
        return BitmapFactory.decodeFile(file.absolutePath)
            .also { Log.d(TAG, "Loaded panorama $id (${file.name}): ${it?.width}×${it?.height}") }
    }

    // ── Glassio sidecar ───────────────────────────────────────────────────────

    /**
     * Write a `.glassio` ZIP sidecar alongside the panorama JPEG for [id].
     *
     * Format (rename to .zip to inspect in any archive manager):
     *   meta.json  — { version, panoramaWidth, frameWidth, frameHeight, keyframes:[{file,angleDeg}] }
     *   kf_000.jpg — full-resolution keyframe JPEG (90% quality)
     *   kf_001.jpg …
     *
     * Blocking — call on a background thread.
     */
    fun saveGlassio(id: String, keyframes: List<Keyframe>, panoramaWidth: Int) {
        if (keyframes.isEmpty()) return
        val jpeg = findJpegFileForId(id) ?: run {
            Log.w(TAG, "saveGlassio: no JPEG found for id=$id")
            return
        }
        val file = File(dir, "${jpeg.nameWithoutExtension}.glassio")
        try {
            ZipOutputStream(FileOutputStream(file)).use { zip ->
                val first = keyframes.first().bitmap
                val meta  = JSONObject().apply {
                    put("version",      1)
                    put("panoramaWidth", panoramaWidth)
                    put("frameWidth",   first.width)
                    put("frameHeight",  first.height)
                    put("keyframes", JSONArray().also { arr ->
                        keyframes.forEachIndexed { i, kf ->
                            arr.put(JSONObject().apply {
                                put("file",     "kf_${"%03d".format(i)}.jpg")
                                put("angleDeg", kf.angleDeg.toDouble())
                            })
                        }
                    })
                }
                zip.putNextEntry(ZipEntry("meta.json"))
                zip.write(meta.toString(2).toByteArray(Charsets.UTF_8))
                zip.closeEntry()

                keyframes.forEachIndexed { i, kf ->
                    zip.putNextEntry(ZipEntry("kf_${"%03d".format(i)}.jpg"))
                    kf.bitmap.compress(Bitmap.CompressFormat.JPEG, 90, zip)
                    zip.closeEntry()
                }
            }
            Log.d(TAG, "Saved glassio ${file.name}: ${keyframes.size} keyframes, ${file.length() / 1024} KB")
        } catch (e: Exception) {
            Log.e(TAG, "saveGlassio failed for $id: ${e.message}")
            file.delete()
        }
    }

    /**
     * Load the `.glassio` sidecar for [id], or return null if it does not exist.
     *
     * Returned bitmaps are freshly decoded and owned by the caller (pass to
     * [PanoramaProcessor.loadAndAnalyzePanorama] which recycles on the next reset).
     *
     * Blocking — call on a background thread.
     */
    fun loadGlassio(id: String): List<Keyframe>? {
        val glassio = findGlassioFileForId(id) ?: return null
        return try {
            val bitmaps = mutableMapOf<String, Bitmap>()
            var kfArray: org.json.JSONArray? = null

            ZipInputStream(glassio.inputStream()).use { zip ->
                var entry = zip.nextEntry
                while (entry != null) {
                    when {
                        entry.name == "meta.json" ->
                            kfArray = JSONObject(zip.readBytes().toString(Charsets.UTF_8))
                                .getJSONArray("keyframes")
                        entry.name.endsWith(".jpg") -> {
                            val bytes = zip.readBytes()
                            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                                ?.let { bitmaps[entry!!.name] = it }
                        }
                    }
                    zip.closeEntry()
                    entry = zip.nextEntry
                }
            }

            val arr = kfArray ?: run {
                Log.w(TAG, "loadGlassio $id: missing meta.json")
                bitmaps.values.forEach { it.recycle() }
                return null
            }

            val keyframes = (0 until arr.length()).mapNotNull { i ->
                val obj    = arr.getJSONObject(i)
                val bitmap = bitmaps[obj.getString("file")] ?: return@mapNotNull null
                Keyframe(
                    bitmap           = bitmap,
                    thumbnail        = bitmap,   // same ref; state.reset() checks isRecycled
                    angleDeg         = obj.getDouble("angleDeg").toFloat(),
                    verticalOffsetPx = 0f,
                    homography       = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
                )
            }

            // Recycle any bitmaps not mapped to a valid keyframe entry.
            val used = keyframes.mapTo(mutableSetOf()) { it.bitmap }
            bitmaps.values.filter { it !in used }.forEach { it.recycle() }

            if (keyframes.isEmpty()) {
                Log.w(TAG, "loadGlassio $id: no valid keyframes decoded")
                return null
            }

            Log.d(TAG, "Loaded glassio $id: ${keyframes.size} keyframes")
            keyframes
        } catch (e: Exception) {
            Log.e(TAG, "loadGlassio failed for $id: ${e.message}")
            null
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────────

    private fun findJpegFileForId(id: String): File? =
        dir.listFiles { f ->
            f.isFile && f.name.lowercase().endsWith(".jpg") && !f.name.endsWith("_thumb.jpg")
        }?.firstOrNull { file ->
            val named = NAMED_PATTERN.matchEntire(file.name)
            val ts    = named?.groupValues?.get(1)?.toLongOrNull() ?: file.lastModified()
            ts.toString() == id
        }

    private fun findGlassioFileForId(id: String): File? {
        val jpeg = findJpegFileForId(id) ?: return null
        return File(dir, "${jpeg.nameWithoutExtension}.glassio").takeIf { it.exists() }
    }
}
