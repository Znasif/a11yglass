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
     * Blocking — call on a background thread.
     */
    fun listSaved(): List<SavedPanorama> {
        val thumbPattern = Regex("""panorama_(\d+)_n(\d+)_thumb\.jpg""")
        return dir.listFiles()
            ?.mapNotNull { file ->
                val match = thumbPattern.matchEntire(file.name) ?: return@mapNotNull null
                val ts = match.groupValues[1].toLongOrNull() ?: return@mapNotNull null
                val n  = match.groupValues[2].toIntOrNull()  ?: return@mapNotNull null
                val mainFile = File(dir, "panorama_${ts}_n${n}.jpg")
                if (!mainFile.exists()) return@mapNotNull null
                val thumb = BitmapFactory.decodeFile(file.absolutePath)
                SavedPanorama(
                    id = ts.toString(),
                    timestamp = ts,
                    nodeCount = n,
                    filePath  = mainFile.absolutePath,
                    thumbnailBitmap = thumb,
                )
            }
            ?.sortedByDescending { it.timestamp }
            ?: emptyList()
    }

    /**
     * Load the full-resolution bitmap for [id].
     * Blocking — call on a background thread.
     */
    fun load(id: String): Bitmap? {
        val file = dir.listFiles()
            ?.firstOrNull { it.name.startsWith("panorama_${id}_n") && !it.name.endsWith("_thumb.jpg") }
            ?: return null
        return BitmapFactory.decodeFile(file.absolutePath)
            .also { Log.d(TAG, "Loaded panorama $id: ${it?.width}×${it?.height}") }
    }
}
