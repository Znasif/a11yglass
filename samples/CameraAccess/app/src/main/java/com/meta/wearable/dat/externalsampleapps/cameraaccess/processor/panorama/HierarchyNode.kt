package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

/**
 * One region in the pre-derived panorama hierarchy.
 *
 * Level-0 = root (full panorama).
 * Level-1 = one node per keyframe (the angular slice that keyframe covers).
 * Level-2 (future) = sub-object detection within each keyframe crop.
 *
 * @param label               Display label; v1 = angle string, v2 = FastVLM caption.
 * @param angleDeg            Angular centre in panorama space (degrees).
 * @param panoramaXFraction   Normalised 0–1 horizontal centre in the stitched bitmap.
 * @param panoramaWidthFraction Normalised 0–1 width in the stitched bitmap.
 * @param keyframeIndex       Index into state.keyframes for this node's source frame.
 * @param description         Filled async (FastVLM) or left as angle label for v1.
 * @param children            Sub-component nodes (Level-2+, currently unused).
 */
data class HierarchyNode(
    val label: String,
    val angleDeg: Float,
    val panoramaXFraction: Float,
    val panoramaWidthFraction: Float,
    val keyframeIndex: Int,
    val description: String = "",
    val children: List<HierarchyNode> = emptyList(),
)
