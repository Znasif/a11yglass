package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF

/**
 * Stateless Canvas renderer for the Reality Proxy overlay.
 *
 * Render pipeline (all coordinates in output-bitmap space):
 *  1. Black background.
 *  2. Fit the stitched panorama into the frame with letterboxing.
 *  3. Draw grey-outline rectangles + angle labels for each hierarchy node.
 *  4. Compute the live-feed cutout window:
 *       horizontal width  = POINTING_FRACTION × camera-hFOV / panoramaSpan × drawW
 *       vertical   height = POINTING_FRACTION × drawH
 *     Both axes use the same fraction so the window is geometrically consistent
 *     with the centre-crop of the live frame (portrait camera: hFOV ≈ 65°,
 *     vFOV ≈ 97° — the POINTING_FRACTION scales each independently).
 *  5. Crop the centre POINTING_FRACTION of the live frame and draw it into the
 *     cutout window with aspect-ratio-preserving scaling (letterbox / pillarbox).
 *  6. Draw a solid blue border around the cutout window.
 *  7. Draw a small white crosshair at the cutout centre.
 *  8. Redraw the focused node (if any) with a cyan highlight.
 *  9. Draw a detail panel in the bottom 22 % with label + description.
 *
 * Paint objects are created once and reused across frames.
 */
class RealityProxyRenderer {

    // ── Node outlines ────────────────────────────────────────────────────────
    private val nodePaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.argb(160, 200, 200, 200)
        strokeWidth = 2f
        isAntiAlias = true
    }
    private val nodeLabelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        isAntiAlias = true
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }

    // ── Live-feed cutout border ───────────────────────────────────────────────
    private val cutoutBorderPaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.argb(255, 80, 160, 255)  // solid blue
        strokeWidth = 3f
        isAntiAlias = true
    }

    // ── Crosshair ────────────────────────────────────────────────────────────
    private val crosshairPaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 2f
        isAntiAlias = true
    }

    // ── Focused node highlight ────────────────────────────────────────────────
    private val focusedFillPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(70, 0, 255, 220)
    }
    private val focusedStrokePaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.argb(220, 0, 255, 220)
        strokeWidth = 3f
        isAntiAlias = true
    }

    // ── Detail panel ─────────────────────────────────────────────────────────
    private val panelPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(180, 0, 0, 0)
    }
    private val detailLabelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 44f
        isFakeBoldText = true
        isAntiAlias = true
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }
    private val detailDescPaint = Paint().apply {
        color = Color.argb(210, 220, 220, 220)
        textSize = 30f
        isAntiAlias = true
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }

    private val PANEL_FRACTION = 0.22f

    /**
     * Composite the panorama + live-feed cutout + hierarchy overlay.
     *
     * @param frame           Live camera frame — used for output dimensions and the
     *                        centre crop drawn inside the cutout window.
     * @param panorama        Stitched panorama bitmap from Phase 2.
     * @param nodes           Pre-built hierarchy nodes (semantic labels from Florence,
     *                        or fallback angle labels).
     * @param currentAngleDeg Estimated angle from PanoramaLocalizer.
     * @param minAngleDeg     Minimum angle in the panorama.
     * @param maxAngleDeg     Maximum angle in the panorama.
     * @param focusedNode     Node under the live-frame centre, or null.
     */
    fun render(
        frame: Bitmap,
        panorama: Bitmap,
        nodes: List<HierarchyNode>,
        currentAngleDeg: Float,
        minAngleDeg: Float,
        maxAngleDeg: Float,
        focusedNode: HierarchyNode?,
    ): Bitmap {
        // Start with a blank black canvas — do NOT copy the live frame as base,
        // so the panorama is always the visual background.
        val output = Bitmap.createBitmap(frame.width, frame.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawColor(Color.BLACK)

        val outW = output.width.toFloat()
        val outH = output.height.toFloat()

        // ── 1. Fit panorama with letterbox ───────────────────────────────────
        val scale   = minOf(outW / panorama.width, outH / panorama.height)
        val drawW   = panorama.width  * scale
        val drawH   = panorama.height * scale
        val offsetX = (outW - drawW) / 2f
        val offsetY = (outH - drawH) / 2f
        canvas.drawBitmap(panorama, null,
            RectF(offsetX, offsetY, offsetX + drawW, offsetY + drawH), null)

        val span = (maxAngleDeg - minAngleDeg).coerceAtLeast(1f)

        // ── 2. Hierarchy node outlines ───────────────────────────────────────
        for (node in nodes) {
            if (node == focusedNode) continue
            val (left, right) = nodeScreenRect(node, offsetX, drawW)
            canvas.drawRect(RectF(left, offsetY, right, offsetY + drawH), nodePaint)
            canvas.drawText(node.label, left + 8f, offsetY + 40f, nodeLabelPaint)
        }

        // ── 3. Live-feed cutout window ────────────────────────────────────────
        //
        // Horizontal: POINTING_FRACTION of the camera's horizontal FOV, mapped
        //             to panorama-display pixels.
        // Vertical:   POINTING_FRACTION of drawH — matches the same fraction of
        //             the frame height that the crop covers in the live image.
        val hFov         = cameraHFovDeg(frame.width, frame.height)
        val cutoutFovDeg = hFov * POINTING_FRACTION
        val fovCenterX   = offsetX + (currentAngleDeg - minAngleDeg) / span * drawW
        val cutoutHalfW  = (cutoutFovDeg / span * drawW / 2f).coerceAtLeast(16f)
        val cutoutHalfH  = POINTING_FRACTION * drawH / 2f
        val panoCenterY  = offsetY + drawH / 2f

        val cutoutRect = RectF(
            fovCenterX  - cutoutHalfW,
            panoCenterY - cutoutHalfH,
            fovCenterX  + cutoutHalfW,
            panoCenterY + cutoutHalfH,
        )

        // Centre-crop POINTING_FRACTION of the live frame
        val cropW = (frame.width  * POINTING_FRACTION).toInt().coerceAtLeast(1)
        val cropH = (frame.height * POINTING_FRACTION).toInt().coerceAtLeast(1)
        val cropX = (frame.width  - cropW) / 2
        val cropY = (frame.height - cropH) / 2
        val liveCrop = Bitmap.createBitmap(frame, cropX, cropY, cropW, cropH)

        // Fit crop into cutout preserving its aspect ratio
        val cropAspect   = cropW.toFloat() / cropH
        val cutoutAspect = cutoutRect.width() / cutoutRect.height().coerceAtLeast(1f)
        val destRect = if (cropAspect > cutoutAspect) {
            // Crop is wider → fit width, letterbox vertically
            val h = cutoutRect.width() / cropAspect
            RectF(cutoutRect.left,          panoCenterY - h / 2f,
                  cutoutRect.right,         panoCenterY + h / 2f)
        } else {
            // Crop is taller → fit height, pillarbox horizontally
            val w = cutoutRect.height() * cropAspect
            RectF(fovCenterX - w / 2f, cutoutRect.top,
                  fovCenterX + w / 2f, cutoutRect.bottom)
        }
        canvas.drawBitmap(liveCrop, null, destRect, null)
        liveCrop.recycle()

        // Blue border
        canvas.drawRect(cutoutRect, cutoutBorderPaint)

        // ── 4. Crosshair at cutout centre ────────────────────────────────────
        val arm = 14f
        canvas.drawLine(fovCenterX - arm, panoCenterY, fovCenterX + arm, panoCenterY, crosshairPaint)
        canvas.drawLine(fovCenterX, panoCenterY - arm, fovCenterX, panoCenterY + arm, crosshairPaint)

        // ── 5. Focused node highlight ────────────────────────────────────────
        focusedNode?.let { node ->
            val (left, right) = nodeScreenRect(node, offsetX, drawW)
            val rect = RectF(left, offsetY, right, offsetY + drawH)
            canvas.drawRect(rect, focusedFillPaint)
            canvas.drawRect(rect, focusedStrokePaint)
            canvas.drawText(node.label, left + 8f, offsetY + 40f, nodeLabelPaint)
        }

        // ── 6. Detail panel ──────────────────────────────────────────────────
        val panelH   = outH * PANEL_FRACTION
        val panelTop = outH - panelH
        canvas.drawRect(RectF(0f, panelTop, outW, outH), panelPaint)

        if (focusedNode != null) {
            canvas.drawText(focusedNode.label, 24f, panelTop + 52f, detailLabelPaint)
            val desc = when {
                focusedNode.description.isNotEmpty() -> focusedNode.description
                else -> "${focusedNode.angleDeg.toInt()}° — tap to describe (coming soon)"
            }
            canvas.drawText(desc, 24f, panelTop + 96f, detailDescPaint)
        } else {
            canvas.drawText(
                "Look at a region to see details",
                24f, panelTop + 64f, detailDescPaint
            )
        }

        return output
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private fun nodeScreenRect(
        node: HierarchyNode,
        offsetX: Float,
        drawW: Float,
    ): Pair<Float, Float> {
        val nodeCenterX   = offsetX + node.panoramaXFraction * drawW
        val halfNodeWidth = node.panoramaWidthFraction * drawW / 2f
        return Pair(nodeCenterX - halfNodeWidth, nodeCenterX + halfNodeWidth)
    }
}
