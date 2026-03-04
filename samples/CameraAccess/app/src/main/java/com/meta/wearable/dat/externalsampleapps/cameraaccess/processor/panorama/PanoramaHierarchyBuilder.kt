package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.RectF
import kotlin.math.abs

/**
 * Builds and queries the Level-1 hierarchy tree from keyframes right after stitching.
 *
 * build() / buildFromFlorence() are called from processorScope (background thread).
 * Once built, nodes is read-only during REALITY_PROXY.
 */
class PanoramaHierarchyBuilder {

    @Volatile private var nodes: List<HierarchyNode> = emptyList()
    // Cached after build() so findNodeAt() uses the real FOV, not a hardcoded guess.
    @Volatile private var computedFovDeg: Float = 65f

    /**
     * Build a fallback Level-1 hierarchy from keyframes (one node per keyframe,
     * labelled by angle). Used when Florence is unavailable or returns nothing.
     *
     * @param keyframes    Accepted keyframes (sorted by angleDeg).
     * @param panoramaWidth Width of the stitched bitmap in px.
     */
    fun build(keyframes: List<Keyframe>, panoramaWidth: Int): List<HierarchyNode> {
        if (keyframes.isEmpty()) return emptyList()

        val sampleBmp = keyframes.first().bitmap
        computedFovDeg = cameraHFovDeg(sampleBmp.width, sampleBmp.height)

        val minAngle  = keyframes.minOf { it.angleDeg }
        val maxAngle  = keyframes.maxOf { it.angleDeg }
        val totalSpan = (maxAngle - minAngle).coerceAtLeast(computedFovDeg)

        return keyframes.mapIndexed { index, kf ->
            HierarchyNode(
                label                 = "${kf.angleDeg.toInt()}°",
                angleDeg              = kf.angleDeg,
                panoramaXFraction     = (kf.angleDeg - minAngle) / totalSpan,
                panoramaWidthFraction = computedFovDeg / totalSpan,
                keyframeIndex         = index,
            )
        }.sortedBy { it.angleDeg }
    }

    /**
     * Build a semantic hierarchy from Florence-2 dense region captions.
     *
     * [regions] is a list of (bbox-in-panorama-px, label) pairs. The bbox's
     * horizontal centre maps to an angle; width maps to a panoramaWidthFraction.
     *
     * @param regions       Florence regions (bbox in panorama pixel coordinates).
     * @param panoramaWidth  Width of the stitched panorama bitmap in px.
     * @param minAngleDeg   Minimum angle in the panorama sweep.
     * @param maxAngleDeg   Maximum angle in the panorama sweep.
     * @param fallbackFovDeg Horizontal FOV, used to seed computedFovDeg for findNodeAt().
     */
    fun buildFromFlorence(
        regions: List<Pair<RectF, String>>,
        panoramaWidth: Int,
        minAngleDeg: Float,
        maxAngleDeg: Float,
        fallbackFovDeg: Float,
    ): List<HierarchyNode> {
        if (regions.isEmpty()) return emptyList()
        computedFovDeg = fallbackFovDeg
        val span = (maxAngleDeg - minAngleDeg).coerceAtLeast(1f)

        return regions.mapIndexed { index, (bbox, label) ->
            val xFraction     = (bbox.centerX() / panoramaWidth).coerceIn(0f, 1f)
            val widthFraction = (bbox.width()   / panoramaWidth).coerceIn(0.01f, 1f)
            HierarchyNode(
                label                 = label,
                angleDeg              = minAngleDeg + xFraction * span,
                panoramaXFraction     = xFraction,
                panoramaWidthFraction = widthFraction,
                keyframeIndex         = index,
            )
        }.sortedBy { it.angleDeg }
    }

    /** Swap in the completed node list atomically. */
    fun setNodes(n: List<HierarchyNode>) { nodes = n }

    /**
     * Return the node whose angular centre is closest to [angleDeg] and within
     * half a FOV of it. Ties broken by closest centre distance.
     */
    fun findNodeAt(angleDeg: Float): HierarchyNode? {
        val halfFov = computedFovDeg / 2f
        return nodes
            .filter { abs(it.angleDeg - angleDeg) < halfFov }
            .minByOrNull { abs(it.angleDeg - angleDeg) }
    }

    fun clear() {
        nodes = emptyList()
        computedFovDeg = 65f
    }
}
