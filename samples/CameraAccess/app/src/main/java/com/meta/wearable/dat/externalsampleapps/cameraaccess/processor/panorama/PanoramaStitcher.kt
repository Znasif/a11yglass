package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.Bitmap
import android.graphics.Bitmap.Config.ARGB_8888
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log

/**
 * Phase-2 post-hoc stitcher.
 *
 * Called once after the user stops sweeping. Composites each keyframe onto a
 * shared canvas using **angle-based pure X-translation** rather than chaining
 * inverse homographies.
 *
 * ## Why not H-chain stitching?
 * Chaining M_i = M_{i-1} × inv(H_{i-1→i}) accumulates perspective distortion
 * and vertical drift over many frames: a 10-frame sweep routinely inflates the
 * estimated canvas to 8192×8192 (268 MB), which Android's hardware canvas
 * rejects with "Canvas: trying to draw too large bitmap".
 *
 * ## Translation approach
 * Each keyframe already carries `angleDeg`, the accumulated horizontal angle
 * measured from the first frame.  Converting angle → pixels gives a stable
 * X-offset with no drift:
 *
 *   offsetX(i) = (kf.angleDeg − minAngle) × (frameWidth / CAMERA_FOV_DEGREES)
 *
 * Canvas width  = (maxAngle − minAngle) × pxPerDeg + frameWidth
 * Canvas height = frameHeight  (always exactly one frame tall; no drift)
 *
 * Compositing uses painter's algorithm (left-to-right order = earlier keyframes
 * drawn first, later ones on top at the seam).
 *
 * Threading: Must be called from a background thread (Dispatchers.Default).
 */
class PanoramaStitcher {

    companion object {
        private const val TAG = "PanoramaStitcher"
        // Absolute upper bound on canvas width in case CAMERA_FOV_DEGREES is
        // wrong or the user sweeps more than a full rotation.
        private const val MAX_CANVAS_WIDTH = 8192
    }

    /**
     * Stitch all keyframes into a single panorama bitmap.
     *
     * @param keyframes Ordered list of accepted keyframes (sorted by angleDeg).
     * @return          Stitched bitmap, or null on failure / fewer than 2 frames.
     */
    fun stitch(keyframes: List<Keyframe>): Bitmap? {
        if (keyframes.isEmpty()) return null
        if (keyframes.size < 2) {
            Log.d(TAG, "Only one keyframe — returning copy directly")
            return keyframes[0].bitmap.copy(ARGB_8888, false)
        }

        val fW = keyframes[0].bitmap.width
        val fH = keyframes[0].bitmap.height
        val pxPerDeg = fW.toFloat() / CAMERA_FOV_DEGREES

        // ── 1. Compute canvas size from angular span ─────────────────────────
        val minAngle = keyframes.minOf { it.angleDeg }
        val maxAngle = keyframes.maxOf { it.angleDeg }
        val spanDeg  = maxAngle - minAngle

        val canvasW = (spanDeg * pxPerDeg + fW).toInt().coerceIn(fW, MAX_CANVAS_WIDTH)
        val canvasH = fH   // horizontal panorama is always exactly one frame tall

        Log.d(TAG, "Stitching ${keyframes.size} frames onto ${canvasW}×${canvasH} canvas " +
              "(${String.format("%.1f", spanDeg)}° span)")

        // ── 2. Composite via pure X-translation ──────────────────────────────
        return try {
            val output = Bitmap.createBitmap(canvasW, canvasH, ARGB_8888)
            val canvas = Canvas(output)
            val paint  = Paint(Paint.FILTER_BITMAP_FLAG)

            for (kf in keyframes) {
                val offsetX = (kf.angleDeg - minAngle) * pxPerDeg
                val m = Matrix()
                m.setTranslate(offsetX, 0f)
                canvas.drawBitmap(kf.bitmap, m, paint)
            }

            Log.d(TAG, "Stitch complete: ${canvasW}×${canvasH}")
            output
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM during stitch (${canvasW}×${canvasH})")
            null
        } catch (e: Exception) {
            Log.e(TAG, "Stitch failed: ${e.message}", e)
            null
        }
    }

    fun release() { /* stateless — nothing to release */ }
}
