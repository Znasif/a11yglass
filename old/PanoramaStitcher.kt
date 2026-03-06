package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.Bitmap
import android.graphics.Bitmap.Config.ARGB_8888
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import android.graphics.Shader
import android.util.Log

/**
 * Phase-2 post-hoc stitcher.
 *
 * Called once after the user stops sweeping. Composites each keyframe onto a
 * shared canvas in two passes:
 *
 * ## Pass 1 – Placement with X+Y offsets
 * Each keyframe is placed using:
 *   - X = (angleDeg − minAngle) × pxPerDeg  (from accumulated horizontal H shifts)
 *   - Y = verticalOffsetPx − minY            (from accumulated vertical H shifts)
 *
 * This corrects the intermittent vertical misalignment that occurs when the
 * camera tilts slightly during the sweep, without requiring extra FeatureTracker
 * inference. Painter's algorithm (left→right order) means later keyframes
 * overwrite earlier ones in overlap regions.
 *
 * ## Pass 2 – Seam blending
 * For each pair of consecutive overlapping frames, the LEFT frame is re-drawn
 * in the overlap zone with a horizontal alpha gradient (1 → 0 from the seam's
 * left edge to its right edge). This fades the left frame back over the right
 * frame that covers it after Pass 1, giving a smooth linear cross-fade at each
 * seam instead of a hard cut.
 *
 * Threading: Must be called from a background thread (Dispatchers.Default).
 */
class PanoramaStitcher {

    companion object {
        private const val TAG = "PanoramaStitcher"
        // Canvas width cap: even if FOV estimate is wrong or the user sweeps
        // more than 360°, the result stays within a drawable size.
        private const val MAX_CANVAS_WIDTH = 8192
        // Cap vertical range at 50 % of one frame height; larger values are
        // artefacts of perspective distortion in long sweeps, not real tilt.
        private const val MAX_Y_FRACTION   = 0.5f
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
        val pxPerDeg = fW.toFloat() / cameraHFovDeg(fW, fH)

        // ── 1. Compute canvas bounds ─────────────────────────────────────────
        val minAngle = keyframes.minOf { it.angleDeg }
        val maxAngle = keyframes.maxOf { it.angleDeg }
        val minY     = keyframes.minOf { it.verticalOffsetPx }
        val maxY     = keyframes.maxOf { it.verticalOffsetPx }
        val yRange   = (maxY - minY).coerceAtMost(fH * MAX_Y_FRACTION)

        val canvasW = ((maxAngle - minAngle) * pxPerDeg + fW).toInt()
            .coerceIn(fW, MAX_CANVAS_WIDTH)
        val canvasH = (fH + yRange).toInt().coerceAtLeast(fH)

        Log.d(TAG, "Stitching ${keyframes.size} frames onto ${canvasW}×${canvasH} " +
              "(${"%.1f".format(maxAngle - minAngle)}° span, Y-range: ${"%.1f".format(yRange)}px)")

        // ── 2. Compute per-frame canvas positions ────────────────────────────
        // Normalise Y so the topmost frame starts at canvas row 0.
        data class Pos(val x: Float, val y: Float)
        val positions = keyframes.map { kf ->
            Pos(
                x = (kf.angleDeg - minAngle) * pxPerDeg,
                y = (kf.verticalOffsetPx - minY).coerceAtMost(yRange)
            )
        }

        return try {
            val output = Bitmap.createBitmap(canvasW, canvasH, ARGB_8888)
            val canvas = Canvas(output)
            val paint  = Paint(Paint.FILTER_BITMAP_FLAG)

            // ── Pass 1: draw all frames with X+Y offsets (painter's algorithm) ──
            for (i in keyframes.indices) {
                val (x, y) = positions[i]
                val m = Matrix()
                m.setTranslate(x, y)
                canvas.drawBitmap(keyframes[i].bitmap, m, paint)
            }

            // ── Pass 2: seam blending ────────────────────────────────────────
            // For each consecutive pair, re-draw the LEFT frame in the overlap
            // zone with a gradient alpha (1 at seamLeft → 0 at seamRight).
            // After Pass 1 the right frame dominates the overlap; this fades
            // the left frame back in to create a smooth cross-fade.
            for (i in 0 until keyframes.size - 1) {
                val (xLeft, yLeft) = positions[i]
                val (xRight, _)    = positions[i + 1]

                val seamLeft  = xRight                   // left edge of overlap zone
                val seamRight = xLeft + fW               // right edge of overlap zone
                if (seamRight <= seamLeft) continue      // gap between frames — skip

                val layerRect = RectF(
                    seamLeft,
                    0f,
                    seamRight.coerceAtMost(canvasW.toFloat()),
                    canvasH.toFloat()
                )

                // saveLayer isolates the left-frame pixels so DST_IN only
                // affects them, not the rest of the canvas.
                val count = canvas.saveLayer(layerRect, null)

                // Draw the left frame at its position (only the overlap portion
                // is visible because saveLayer clips to layerRect).
                val m = Matrix()
                m.setTranslate(xLeft, yLeft)
                canvas.drawBitmap(keyframes[i].bitmap, m, null)

                // Gradient mask: opaque at seamLeft, transparent at seamRight.
                // DST_IN keeps DST (left frame) multiplied by SRC alpha (gradient),
                // producing left_frame_pixel × gradient_alpha on the layer.
                val maskPaint = Paint().apply {
                    shader = LinearGradient(
                        seamLeft, 0f,
                        seamRight, 0f,
                        Color.BLACK, Color.TRANSPARENT,
                        Shader.TileMode.CLAMP
                    )
                    xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
                }
                canvas.drawRect(layerRect, maskPaint)
                canvas.restoreToCount(count)
                // The restored layer composites onto the main canvas via SRC_OVER,
                // blending the faded left frame over the right frame from Pass 1.
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
