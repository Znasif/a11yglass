package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import kotlin.math.abs

/**
 * Builds and queries the Level-1 hierarchy tree from keyframes right after stitching.
 *
 * build() is called from processorScope (background thread) immediately after Phase 2.
 * Once built, nodes is read-only during REALITY_PROXY.
 *
 * v2 (deferred): After build() returns, iterate nodes, write each keyframe crop to a
 * temp file, call FastVLM Engine, and update node.description progressively.
 */
class PanoramaHierarchyBuilder {

    @Volatile private var nodes: List<HierarchyNode> = emptyList()

    /**
     * Build a Level-1 hierarchy from keyframes.
     *
     * Algorithm:
     *   totalSpan = max(maxAngle - minAngle, CAMERA_FOV_DEGREES)  // guard zero-span
     *   For each keyframe kf:
     *     xFraction     = (kf.angleDeg - minAngle) / totalSpan
     *     widthFraction = CAMERA_FOV_DEGREES / totalSpan
     *     label         = "${kf.angleDeg.toInt()}°"
     *
     * @param keyframes    Accepted keyframes (sorted by angleDeg).
     * @param panoramaWidth Width of the stitched bitmap in px (for future Level-2 use).
     * @return             List of Level-1 nodes sorted by angleDeg.
     */
    fun build(keyframes: List<Keyframe>, panoramaWidth: Int): List<HierarchyNode> {
        if (keyframes.isEmpty()) return emptyList()

        val minAngle  = keyframes.minOf { it.angleDeg }
        val maxAngle  = keyframes.maxOf { it.angleDeg }
        // Clamp to at least one FOV worth of span to avoid division by zero
        val totalSpan = (maxAngle - minAngle).coerceAtLeast(CAMERA_FOV_DEGREES)

        return keyframes.mapIndexed { index, kf ->
            HierarchyNode(
                label                = "${kf.angleDeg.toInt()}°",
                angleDeg             = kf.angleDeg,
                panoramaXFraction    = (kf.angleDeg - minAngle) / totalSpan,
                panoramaWidthFraction = CAMERA_FOV_DEGREES / totalSpan,
                keyframeIndex        = index,
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
        val halfFov = CAMERA_FOV_DEGREES / 2f
        return nodes
            .filter { abs(it.angleDeg - angleDeg) < halfFov }
            .minByOrNull { abs(it.angleDeg - angleDeg) }
    }

    fun clear() { nodes = emptyList() }
}
