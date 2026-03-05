package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.io.File
import java.io.FileOutputStream

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
        val namedPattern = Regex("""panorama_(\d+)_n(\d+)\.jpg""")
        val allJpegs = dir.listFiles { f ->
            f.isFile && f.name.lowercase().endsWith(".jpg") && !f.name.endsWith("_thumb.jpg")
        } ?: return emptyList()

        return allJpegs.mapNotNull { file ->
            val named = namedPattern.matchEntire(file.name)
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
        // Find the main file whose derived timestamp matches [id], then delete it
        // and its thumbnail (which shares the same stem + "_thumb.jpg").
        val namedPattern = Regex("""panorama_(\d+)_n(\d+)\.jpg""")
        dir.listFiles { f ->
            f.isFile && f.name.lowercase().endsWith(".jpg") && !f.name.endsWith("_thumb.jpg")
        }?.forEach { file ->
            val named = namedPattern.matchEntire(file.name)
            val ts = named?.groupValues?.get(1)?.toLongOrNull() ?: file.lastModified()
            if (ts.toString() == id) {
                val thumbFile = File(dir, "${file.nameWithoutExtension}_thumb.jpg")
                file.delete()
                if (thumbFile.exists()) thumbFile.delete()
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
        val namedPattern = Regex("""panorama_(\d+)_n(\d+)\.jpg""")
        val file = dir.listFiles { f ->
            f.isFile && f.name.lowercase().endsWith(".jpg") && !f.name.endsWith("_thumb.jpg")
        }?.firstOrNull { file ->
            val named = namedPattern.matchEntire(file.name)
            val ts = named?.groupValues?.get(1)?.toLongOrNull() ?: file.lastModified()
            ts.toString() == id
        } ?: return null
        return BitmapFactory.decodeFile(file.absolutePath)
            .also { Log.d(TAG, "Loaded panorama $id (${file.name}): ${it?.width}×${it?.height}") }
    }
}
