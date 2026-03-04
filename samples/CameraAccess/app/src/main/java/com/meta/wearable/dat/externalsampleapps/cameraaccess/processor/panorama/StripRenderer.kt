package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.floor

/**
 * Stateless Canvas renderer for the live panorama strip preview.
 *
 * Strip coordinate system:
 *   stripX(deg) = frameWidth/2 + (deg – currentAngleDeg) × PX_PER_DEG
 * → current angle is always horizontally centred.
 *
 * Elements drawn (bottom of frame):
 *  1. Dark semi-transparent background
 *  2. Grey fill for visible angular range [currentAngle ± STRIP_VISIBLE_RANGE]
 *  3. Degree tick marks every 15°
 *  4. Keyframe thumbnails at their angular positions
 *  5. Red gap markers where consecutive keyframes exceed GAP_WARN_DEGREES
 *  6. Blue FOV rectangle [currentAngle ± FOV/2] + white border
 *  7. Status text (frame count, current angle)
 *  8. If stitchedResult != null: show stitched panorama instead of thumbnails
 *  9. If idle (not capturing, no keyframes): "Camera button to start" hint
 */
class StripRenderer {

    // ── Paints (reused across draw calls) ───────────────────────────────────

    private val bgPaint = Paint().apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val rangePaint = Paint().apply {
        color = Color.argb(55, 140, 140, 140)
        style = Paint.Style.FILL
    }

    private val tickPaint = Paint().apply {
        color = Color.argb(120, 255, 255, 255)
        strokeWidth = 1.5f
        style = Paint.Style.STROKE
        isAntiAlias = false
    }

    private val fovFillPaint = Paint().apply {
        color = Color.argb(90, 80, 140, 255)
        style = Paint.Style.FILL
    }

    private val fovBorderPaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 2f
        style = Paint.Style.STROKE
    }

    private val gapLinePaint = Paint().apply {
        color = Color.RED
        strokeWidth = 3f
        style = Paint.Style.STROKE
    }

    private val gapLabelPaint = Paint().apply {
        color = Color.RED
        textSize = 28f
        isFakeBoldText = true
        isAntiAlias = true
    }

    private val statusPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        isAntiAlias = true
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }

    private val hintPaint = Paint().apply {
        color = Color.argb(200, 255, 255, 255)
        textSize = 34f
        isAntiAlias = true
        textAlign = Paint.Align.CENTER
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /**
     * Draw the panorama strip onto [canvas].
     *
     * @param canvas         Target canvas (already a mutable copy of the frame).
     * @param frameWidth     Full frame width in pixels.
     * @param frameHeight    Full frame height in pixels.
     * @param currentAngleDeg Accumulated angular position for the current view.
     * @param keyframes      Accepted keyframes, sorted by angleDeg.
     * @param isCapturing    Whether a sweep session is in progress.
     * @param stitchedResult If non-null, the Phase-2 stitched panorama to preview.
     */
    fun draw(
        canvas: Canvas,
        frameWidth: Int,
        frameHeight: Int,
        currentAngleDeg: Float,
        keyframes: List<Keyframe>,
        isCapturing: Boolean,
        stitchedResult: Bitmap?
    ) {
        val stripH = (frameHeight * STRIP_HEIGHT_FRACTION).toInt()
        val stripTop = (frameHeight - stripH).toFloat()
        val fW = frameWidth.toFloat()
        val fH = frameHeight.toFloat()

        // 1. Dark background
        canvas.drawRect(0f, stripTop, fW, fH, bgPaint)

        // 2. Grey fill for visible angular range
        val rangeLeft  = sx(currentAngleDeg - STRIP_VISIBLE_RANGE, frameWidth, currentAngleDeg)
        val rangeRight = sx(currentAngleDeg + STRIP_VISIBLE_RANGE, frameWidth, currentAngleDeg)
        canvas.drawRect(rangeLeft, stripTop, rangeRight, fH, rangePaint)

        // ── If stitched result available, show it and exit early ─────────────
        if (stitchedResult != null) {
            val destRect = RectF(0f, stripTop, fW, fH)
            canvas.drawBitmap(stitchedResult, null, destRect, null)
            canvas.drawText(
                "Panorama ready — ${keyframes.size} frames",
                20f, stripTop + 36f, statusPaint
            )
            return
        }

        // 3. Degree tick marks every 15°
        drawTicks(canvas, frameWidth, frameHeight, stripTop, stripH, currentAngleDeg)

        // 4. Keyframe thumbnails
        drawThumbnails(canvas, frameWidth, frameHeight, stripTop, currentAngleDeg, keyframes)

        // 5. Red gap markers
        drawGapMarkers(canvas, frameWidth, stripTop, currentAngleDeg, keyframes)

        // 6. FOV rectangle
        val hFov     = cameraHFovDeg(frameWidth, frameHeight)
        val fovLeft  = sx(currentAngleDeg - hFov / 2f, frameWidth, currentAngleDeg)
        val fovRight = sx(currentAngleDeg + hFov / 2f, frameWidth, currentAngleDeg)
        canvas.drawRect(fovLeft, stripTop, fovRight, fH, fovFillPaint)
        canvas.drawRect(fovLeft, stripTop, fovRight, fH, fovBorderPaint)

        // 7. Status label
        val statusText = when {
            isCapturing -> "${keyframes.size} frames | ${"%.0f".format(currentAngleDeg)}° | sweeping…"
            keyframes.isNotEmpty() -> "Stopped: ${keyframes.size} frames | stitching…"
            else -> ""
        }
        if (statusText.isNotEmpty()) {
            canvas.drawText(statusText, 20f, stripTop + 36f, statusPaint)
        }

        // 8. Idle hint
        if (!isCapturing && keyframes.isEmpty()) {
            canvas.drawText(
                "Camera button to start panorama",
                fW / 2f, stripTop + stripH / 2f, hintPaint
            )
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /** Map an angular position to a strip x-coordinate. */
    private fun sx(angleDeg: Float, frameWidth: Int, currentAngleDeg: Float): Float =
        frameWidth / 2f + (angleDeg - currentAngleDeg) * PX_PER_DEG

    private fun drawTicks(
        canvas: Canvas,
        frameWidth: Int,
        frameHeight: Int,
        stripTop: Float,
        stripH: Int,
        currentAngleDeg: Float
    ) {
        val minDeg = currentAngleDeg - STRIP_VISIBLE_RANGE
        val maxDeg = currentAngleDeg + STRIP_VISIBLE_RANGE
        val tickStep = 15f
        var tick = floor((minDeg / tickStep).toDouble()).toFloat() * tickStep
        val fH = frameHeight.toFloat()
        val fW = frameWidth.toFloat()

        while (tick <= maxDeg) {
            val tx = sx(tick, frameWidth, currentAngleDeg)
            if (tx in 0f..fW) {
                val tickLen = if (tick % 30 == 0f) stripH * 0.35f else stripH * 0.18f
                canvas.drawLine(tx, fH - tickLen, tx, fH, tickPaint)
            }
            tick += tickStep
        }
    }

    private fun drawThumbnails(
        canvas: Canvas,
        frameWidth: Int,
        frameHeight: Int,
        stripTop: Float,
        currentAngleDeg: Float,
        keyframes: List<Keyframe>
    ) {
        val fW = frameWidth.toFloat()
        val fH = frameHeight.toFloat()
        for (kf in keyframes) {
            val cx = sx(kf.angleDeg, frameWidth, currentAngleDeg)
            val halfW = kf.thumbnail.width / 2f
            val left  = cx - halfW
            val right = cx + halfW
            if (right < 0f || left > fW) continue  // entirely off-screen
            canvas.drawBitmap(kf.thumbnail, null, RectF(left, stripTop, right, fH), null)
        }
    }

    private fun drawGapMarkers(
        canvas: Canvas,
        frameWidth: Int,
        stripTop: Float,
        currentAngleDeg: Float,
        keyframes: List<Keyframe>
    ) {
        val fW = frameWidth.toFloat()
        for (i in 0 until keyframes.size - 1) {
            val gap = abs(keyframes[i + 1].angleDeg - keyframes[i].angleDeg)
            if (gap > GAP_WARN_DEGREES) {
                val midDeg = (keyframes[i].angleDeg + keyframes[i + 1].angleDeg) / 2f
                val gx = sx(midDeg, frameWidth, currentAngleDeg)
                if (gx in 0f..fW) {
                    canvas.drawLine(gx, stripTop, gx, stripTop + (frameWidth * STRIP_HEIGHT_FRACTION), gapLinePaint)
                    canvas.drawText("gap!", gx + 4f, stripTop + 36f, gapLabelPaint)
                }
            }
        }
    }
}
