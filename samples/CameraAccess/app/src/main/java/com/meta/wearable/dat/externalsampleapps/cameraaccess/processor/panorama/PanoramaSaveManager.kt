package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
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
    val hasGlassio: Boolean = false,
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
                hasGlassio      = File(dir, "${file.nameWithoutExtension}.glassio").exists(),
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
     * Write a `.glassio` v2 ZIP sidecar alongside the panorama JPEG for [id].
     *
     * Contents (rename to .zip to inspect):
     *   meta.json    — v2: keyframes + metadata block + hotspots[]
     *   colorMap.png — panorama-sized ARGB bitmap; each hotspot region painted its solid color
     *   kf_000.jpg … — full-resolution keyframe JPEGs
     *
     * [nodes] may be empty (e.g. Florence still running); in that case only
     * keyframes and an empty hotspots array are written.
     *
     * Blocking — call on a background thread.
     */
    fun saveGlassio(
        id: String,
        keyframes: List<Keyframe>,
        panoramaWidth: Int,
        nodes: List<HierarchyNode> = emptyList(),
        panoramaHeight: Int = 0,
        shortDescription: String = "",
        longDescription: String = "",
        title: String = "",
    ) {
        if (keyframes.isEmpty()) return
        val jpeg = findJpegFileForId(id) ?: run {
            Log.w(TAG, "saveGlassio: no JPEG found for id=$id")
            return
        }
        val file = File(dir, "${jpeg.nameWithoutExtension}.glassio")
        try {
            ZipOutputStream(FileOutputStream(file)).use { zip ->
                val first = keyframes.first().bitmap

                // ── meta.json ──────────────────────────────────────────────
                val meta = JSONObject().apply {
                    put("version",       2)
                    put("panoramaWidth", panoramaWidth)
                    put("panoramaHeight",panoramaHeight)
                    put("frameWidth",    first.width)
                    put("frameHeight",   first.height)
                    put("metadata", JSONObject().apply {
                        put("title",            title)
                        put("shortDescription", shortDescription)
                        put("longDescription",  longDescription)
                        put("lang",             "en-US")
                    })
                    put("keyframes", JSONArray().also { arr ->
                        keyframes.forEachIndexed { i, kf ->
                            arr.put(JSONObject().apply {
                                put("file",     "kf_${"%03d".format(i)}.jpg")
                                put("angleDeg", kf.angleDeg.toDouble())
                            })
                        }
                    })
                    put("hotspots", JSONArray().also { arr ->
                        nodes.forEach { node ->
                            arr.put(JSONObject().apply {
                                put("color", JSONArray().apply {
                                    put(Color.red(node.color))
                                    put(Color.green(node.color))
                                    put(Color.blue(node.color))
                                    put(1)
                                })
                                put("title",                  node.label)
                                put("description",            node.description)
                                put("panoramaXFraction",      node.panoramaXFraction.toDouble())
                                put("panoramaYFraction",      node.panoramaYFraction.toDouble())
                                put("panoramaWidthFraction",  node.panoramaWidthFraction.toDouble())
                                put("panoramaHeightFraction", node.panoramaHeightFraction.toDouble())
                                node.polygonXY?.let { pts ->
                                    put("polygon", JSONArray().also { pa ->
                                        pts.forEach { pa.put(it.toDouble()) }
                                    })
                                }
                            })
                        }
                    })
                }
                zip.putNextEntry(ZipEntry("meta.json"))
                zip.write(meta.toString(2).toByteArray(Charsets.UTF_8))
                zip.closeEntry()

                // ── colorMap.png ───────────────────────────────────────────
                if (nodes.isNotEmpty() && panoramaWidth > 0 && panoramaHeight > 0) {
                    val colorMap = buildColorMap(nodes, panoramaWidth, panoramaHeight)
                    zip.putNextEntry(ZipEntry("colorMap.png"))
                    colorMap.compress(Bitmap.CompressFormat.PNG, 100, zip)
                    zip.closeEntry()
                    colorMap.recycle()
                }

                // ── keyframe JPEGs ─────────────────────────────────────────
                keyframes.forEachIndexed { i, kf ->
                    zip.putNextEntry(ZipEntry("kf_${"%03d".format(i)}.jpg"))
                    kf.bitmap.compress(Bitmap.CompressFormat.JPEG, 90, zip)
                    zip.closeEntry()
                }
            }
            Log.d(TAG, "Saved glassio v2 ${file.name}: ${keyframes.size} kf, " +
                "${nodes.size} hotspots, ${file.length() / 1024} KB")
        } catch (e: Exception) {
            Log.e(TAG, "saveGlassio failed for $id: ${e.message}")
            file.delete()
        }
    }

    /**
     * Paint each node's segmentation polygon (or bbox fallback) onto a
     * [panoramaWidth]×[panoramaHeight] bitmap using its solid [HierarchyNode.color].
     * Background is transparent. Used as colorMap.png for touch hit-testing.
     */
    private fun buildColorMap(
        nodes: List<HierarchyNode>,
        panoramaWidth: Int,
        panoramaHeight: Int,
    ): Bitmap {
        val bmp    = Bitmap.createBitmap(panoramaWidth, panoramaHeight, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        val paint  = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
        val w      = panoramaWidth.toFloat()
        val h      = panoramaHeight.toFloat()

        nodes.forEach { node ->
            paint.color = node.color or (0xFF shl 24).toInt()  // force full opacity
            val pts = node.polygonXY
            if (pts != null && pts.size >= 6) {
                val path = Path()
                path.moveTo(pts[0] * w, pts[1] * h)
                for (i in 2 until pts.size - 1 step 2) path.lineTo(pts[i] * w, pts[i + 1] * h)
                path.close()
                canvas.drawPath(path, paint)
            } else {
                // Bbox fallback
                val cx = node.panoramaXFraction * w
                val cy = node.panoramaYFraction * h
                val hw = node.panoramaWidthFraction  * w / 2f
                val hh = node.panoramaHeightFraction * h / 2f
                canvas.drawRect(cx - hw, cy - hh, cx + hw, cy + hh, paint)
            }
        }
        return bmp
    }

    /**
     * Load the `.glassio` sidecar for [id]. Supports both v1 and v2.
     *
     * Returns a [GlassioData] containing keyframes and (v2 only) hierarchy nodes.
     * Returns null if the sidecar does not exist or cannot be parsed.
     *
     * Returned bitmaps are freshly decoded and owned by the caller.
     * Blocking — call on a background thread.
     */
    data class GlassioData(
        val keyframes: List<Keyframe>,
        val nodes: List<HierarchyNode> = emptyList(),
        val shortDescription: String = "",
        val longDescription: String = "",
        val title: String = "",
    )

    fun loadGlassio(id: String): GlassioData? {
        val glassio = findGlassioFileForId(id) ?: return null
        return try {
            val bitmaps    = mutableMapOf<String, Bitmap>()
            var metaObj: JSONObject? = null

            ZipInputStream(glassio.inputStream()).use { zip ->
                var entry = zip.nextEntry
                while (entry != null) {
                    when {
                        entry.name == "meta.json" ->
                            metaObj = JSONObject(zip.readBytes().toString(Charsets.UTF_8))
                        entry.name.endsWith(".jpg") -> {
                            val bytes = zip.readBytes()
                            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                                ?.let { bitmaps[entry!!.name] = it }
                        }
                        // colorMap.png is not loaded at startup — only needed for hit-testing
                    }
                    zip.closeEntry()
                    entry = zip.nextEntry
                }
            }

            val meta = metaObj ?: run {
                Log.w(TAG, "loadGlassio $id: missing meta.json")
                bitmaps.values.forEach { it.recycle() }
                return null
            }

            val version   = meta.optInt("version", 1)
            val kfArray   = meta.getJSONArray("keyframes")
            val keyframes = (0 until kfArray.length()).mapNotNull { i ->
                val obj    = kfArray.getJSONObject(i)
                val bitmap = bitmaps[obj.getString("file")] ?: return@mapNotNull null
                Keyframe(
                    bitmap           = bitmap,
                    thumbnail        = bitmap,
                    angleDeg         = obj.getDouble("angleDeg").toFloat(),
                    verticalOffsetPx = 0f,
                    homography       = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f),
                )
            }

            val used = keyframes.mapTo(mutableSetOf()) { it.bitmap }
            bitmaps.values.filter { it !in used }.forEach { it.recycle() }

            if (keyframes.isEmpty()) {
                Log.w(TAG, "loadGlassio $id: no valid keyframes decoded")
                return null
            }

            // ── v2: load metadata and hotspots ─────────────────────────────
            var nodes            = emptyList<HierarchyNode>()
            var shortDescription = ""
            var longDescription  = ""
            var title            = ""

            if (version >= 2) {
                val metadataObj = meta.optJSONObject("metadata")
                shortDescription = metadataObj?.optString("shortDescription") ?: ""
                longDescription  = metadataObj?.optString("longDescription")  ?: ""
                title            = metadataObj?.optString("title")            ?: ""

                val hotspotsArr = meta.optJSONArray("hotspots")
                if (hotspotsArr != null) {
                    nodes = (0 until hotspotsArr.length()).mapNotNull { i ->
                        val h   = hotspotsArr.getJSONObject(i)
                        val col = h.optJSONArray("color")
                        val color = if (col != null && col.length() >= 3)
                            Color.rgb(col.getInt(0), col.getInt(1), col.getInt(2))
                        else Color.TRANSPARENT

                        val polyArr = h.optJSONArray("polygon")
                        val polygon = if (polyArr != null && polyArr.length() >= 6) {
                            FloatArray(polyArr.length()) { j -> polyArr.getDouble(j).toFloat() }
                        } else null

                        HierarchyNode(
                            label                 = h.optString("title"),
                            angleDeg              = (h.optDouble("panoramaXFraction", 0.5) * 360).toFloat(),
                            panoramaXFraction     = h.optDouble("panoramaXFraction", 0.5).toFloat(),
                            panoramaYFraction     = h.optDouble("panoramaYFraction", 0.5).toFloat(),
                            panoramaWidthFraction = h.optDouble("panoramaWidthFraction", 0.1).toFloat(),
                            panoramaHeightFraction= h.optDouble("panoramaHeightFraction", 1.0).toFloat(),
                            keyframeIndex         = -1,
                            description           = h.optString("description"),
                            color                 = color,
                            polygonXY             = polygon,
                        )
                    }
                }
            }

            Log.d(TAG, "Loaded glassio v$version $id: ${keyframes.size} kf, ${nodes.size} hotspots")
            GlassioData(keyframes, nodes, shortDescription, longDescription, title)
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
