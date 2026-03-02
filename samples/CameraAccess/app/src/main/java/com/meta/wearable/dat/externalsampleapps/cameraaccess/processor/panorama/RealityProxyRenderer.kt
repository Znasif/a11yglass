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
 *  1. Fit the stitched panorama into the frame with letterboxing.
 *  2. Draw grey-outline rectangles + angle labels for each hierarchy node.
 *  3. Draw a semi-transparent blue FOV rectangle showing the live camera window.
 *  4. Draw a white crosshair at the FOV centre.
 *  5. Redraw the focused node (if any) with a cyan highlight.
 *  6. Draw a detail panel in the bottom 22 % with label + description.
 *
 * Paint objects are created once and reused across frames.
 */
class RealityProxyRenderer {

    // ── Node outlines ────────────────────────────────────────────────────────
    private val nodePaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.argb(160, 200, 200, 200) // grey outline
        strokeWidth = 2f
        isAntiAlias = true
    }
    private val nodeLabelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        isAntiAlias = true
        setShadowLayer(2f, 1f, 1f, Color.BLACK)
    }

    // ── FOV indicator ────────────────────────────────────────────────────────
    private val fovFillPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(60, 80, 160, 255)  // semi-transparent blue
    }
    private val fovBorderPaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.WHITE
        strokeWidth = 2.5f
        isAntiAlias = true
    }

    // ── Crosshair ────────────────────────────────────────────────────────────
    private val crosshairPaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 2.5f
        isAntiAlias = true
    }

    // ── Focused node highlight ────────────────────────────────────────────────
    private val focusedFillPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(70, 0, 255, 220)   // semi-transparent cyan
    }
    private val focusedStrokePaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.argb(220, 0, 255, 220)  // bright cyan
        strokeWidth = 3f
        isAntiAlias = true
    }

    // ── Detail panel ─────────────────────────────────────────────────────────
    private val panelPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(180, 0, 0, 0)      // dark semi-transparent background
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

    // ── Panel height fraction ─────────────────────────────────────────────────
    private val PANEL_FRACTION = 0.22f

    /**
     * Composite the panorama + hierarchy overlay onto a mutable copy of [frame].
     *
     * @param frame           Live camera frame — used only for output dimensions.
     * @param panorama        Stitched panorama bitmap from Phase 2.
     * @param nodes           Pre-built Level-1 hierarchy nodes.
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
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

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

        // Guard: avoid division by zero if panorama spans < 1°
        val span = (maxAngleDeg - minAngleDeg).coerceAtLeast(1f)

        // ── 2. Hierarchy node outlines ───────────────────────────────────────
        for (node in nodes) {
            if (node == focusedNode) continue // drawn later with highlight
            val (left, right) = nodeScreenRect(node, offsetX, drawW, span, minAngleDeg)
            canvas.drawRect(RectF(left, offsetY, right, offsetY + drawH), nodePaint)
            canvas.drawText(node.label, left + 8f, offsetY + 40f, nodeLabelPaint)
        }

        // ── 3. FOV indicator ─────────────────────────────────────────────────
        val fovSpan    = CAMERA_FOV_DEGREES / span * drawW
        val fovCenterX = offsetX + (currentAngleDeg - minAngleDeg) / span * drawW
        val fovRect    = RectF(
            fovCenterX - fovSpan / 2f, offsetY,
            fovCenterX + fovSpan / 2f, offsetY + drawH
        )
        canvas.drawRect(fovRect, fovFillPaint)
        canvas.drawRect(fovRect, fovBorderPaint)

        // ── 4. Crosshair at FOV centre ───────────────────────────────────────
        val crossY = offsetY + drawH / 2f
        val arm    = 40f
        canvas.drawLine(fovCenterX - arm, crossY, fovCenterX + arm, crossY, crosshairPaint)
        canvas.drawLine(fovCenterX, crossY - arm, fovCenterX, crossY + arm, crosshairPaint)

        // ── 5. Focused node highlight ────────────────────────────────────────
        focusedNode?.let { node ->
            val (left, right) = nodeScreenRect(node, offsetX, drawW, span, minAngleDeg)
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

    /** Compute screen-space left/right x-coordinates for a node's outline rect. */
    private fun nodeScreenRect(
        node: HierarchyNode,
        offsetX: Float,
        drawW: Float,
        span: Float,
        minAngleDeg: Float,
    ): Pair<Float, Float> {
        val nodeCenterX    = offsetX + node.panoramaXFraction * drawW
        val halfNodeWidth  = node.panoramaWidthFraction * drawW / 2f
        return Pair(nodeCenterX - halfNodeWidth, nodeCenterX + halfNodeWidth)
    }
}
