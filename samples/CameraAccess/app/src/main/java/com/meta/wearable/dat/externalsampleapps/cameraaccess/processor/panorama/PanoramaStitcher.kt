package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.Bitmap
import android.graphics.Bitmap.Config.ARGB_8888
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log

/**
 * Phase-2 post-hoc stitcher.
 *
 * Called once after the user stops sweeping. Takes all accepted keyframes and
 * their pairwise homographies, chains them into a common canvas space, and
 * composites every frame using simple painter's algorithm (newer frames on top).
 *
 * Threading: Must be called from a background thread (Dispatchers.Default).
 */
class PanoramaStitcher {

    companion object {
        private const val TAG = "PanoramaStitcher"
        private const val MAX_CANVAS_DIM = 8192  // Safety cap to avoid OOM on huge sweeps
    }

    /**
     * Stitch all keyframes into a single panorama bitmap.
     *
     * @param keyframes Ordered list of accepted keyframes (sorted by angleDeg).
     * @return          Stitched bitmap, or null on failure / < 2 frames.
     */
    fun stitch(keyframes: List<Keyframe>): Bitmap? {
        if (keyframes.isEmpty()) return null
        if (keyframes.size < 2) {
            Log.d(TAG, "Only one keyframe — returning copy directly")
            return keyframes[0].bitmap.copy(ARGB_8888, false)
        }

        val fW = keyframes[0].bitmap.width
        val fH = keyframes[0].bitmap.height

        // ── 1. Build cumulative canvas-space matrices ────────────────────────
        // Frame 0 sits at the canvas origin (identity).
        // For frame i > 0, its stored H maps frame(i-1) → frame(i).
        // To paint frame i onto the canvas (frame-0 space) we need:
        //   M_i = M_{i-1} · inv(H_{i-1→i})
        val matrices = ArrayList<Matrix>(keyframes.size)
        matrices.add(Matrix())  // identity for frame 0

        var cumulative = Matrix()
        for (i in 1 until keyframes.size) {
            val H = keyframes[i].homography
            val mH = toAndroidMatrix(H)
            val mHInv = Matrix()
            if (mH.invert(mHInv)) {
                val next = Matrix(cumulative)
                next.postConcat(mHInv)
                cumulative = next
            } else {
                Log.w(TAG, "Could not invert H for frame $i — reusing previous matrix")
                cumulative = Matrix(cumulative)
            }
            matrices.add(Matrix(cumulative))
        }

        // ── 2. Estimate canvas bounds ────────────────────────────────────────
        val bounds = estimateBounds(keyframes, matrices, fW, fH)
        if (bounds.isEmpty) {
            Log.e(TAG, "Empty bounds — aborting stitch")
            return null
        }

        val canvasW = bounds.width().toInt().coerceIn(fW, MAX_CANVAS_DIM)
        val canvasH = bounds.height().toInt().coerceIn(fH, MAX_CANVAS_DIM)
        Log.d(TAG, "Stitching ${keyframes.size} frames onto ${canvasW}×${canvasH} canvas")

        // ── 3. Composite onto canvas ─────────────────────────────────────────
        return try {
            val output = Bitmap.createBitmap(canvasW, canvasH, ARGB_8888)
            val canvas = Canvas(output)
            val paint = Paint(Paint.FILTER_BITMAP_FLAG)

            val offsetX = -bounds.left
            val offsetY = -bounds.top

            for (i in keyframes.indices) {
                val m = Matrix(matrices[i])
                m.postTranslate(offsetX, offsetY)
                canvas.drawBitmap(keyframes[i].bitmap, m, paint)
            }

            Log.d(TAG, "Stitch complete")
            output
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM during stitch — canvas too large? (${canvasW}×${canvasH})")
            null
        } catch (e: Exception) {
            Log.e(TAG, "Stitch failed: ${e.message}", e)
            null
        }
    }

    /**
     * Convert a 3×3 row-major FloatArray homography to an Android [Matrix].
     * Layout: [row0col0, row0col1, row0col2, row1col0, row1col1, row1col2, row2col0, row2col1, row2col2]
     */
    private fun toAndroidMatrix(H: FloatArray): Matrix {
        val m = Matrix()
        m.setValues(floatArrayOf(H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]))
        return m
    }

    /**
     * Project each frame's four corners through its canvas-space matrix and
     * return the axis-aligned bounding rectangle of all projected corners.
     */
    private fun estimateBounds(
        keyframes: List<Keyframe>,
        matrices: List<Matrix>,
        fW: Int,
        fH: Int
    ): RectF {
        val corners = floatArrayOf(
            0f, 0f,
            fW.toFloat(), 0f,
            fW.toFloat(), fH.toFloat(),
            0f, fH.toFloat()
        )

        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE
        var maxY = -Float.MAX_VALUE

        for (i in keyframes.indices) {
            val pts = corners.copyOf()
            matrices[i].mapPoints(pts)
            for (j in 0 until 4) {
                val x = pts[j * 2]
                val y = pts[j * 2 + 1]
                if (x < minX) minX = x
                if (y < minY) minY = y
                if (x > maxX) maxX = x
                if (y > maxY) maxY = y
            }
        }

        return RectF(minX, minY, maxX, maxY)
    }

    fun release() { /* stateless — nothing to release */ }
}
