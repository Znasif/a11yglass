package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

/**
 * One region in the pre-derived panorama hierarchy.
 *
 * Level-0 = root (full panorama).
 * Level-1 = one node per keyframe (the angular slice that keyframe covers).
 * Level-2 (future) = sub-object detection within each keyframe crop.
 *
 * @param label                 Display label; v1 = angle string, v2 = Florence caption.
 * @param angleDeg              Angular centre in panorama space (degrees).
 * @param panoramaXFraction     Normalised 0–1 horizontal centre in the stitched bitmap.
 * @param panoramaWidthFraction Normalised 0–1 width in the stitched bitmap.
 * @param keyframeIndex         Index into state.keyframes for this node's source frame.
 * @param description           Filled by <REGION_TO_DESCRIPTION> after stitching.
 * @param color                 ARGB int used for the overlay fill and colorMap.png.
 * @param polygonXY             Flat [x0,y0,x1,y1,…] in normalised 0–1 panorama fractions,
 *                              from <REGION_TO_SEGMENTATION>. Null = use bbox fallback.
 * @param children              Sub-component nodes (Level-2+, currently unused).
 */
data class HierarchyNode(
    val label: String,
    val angleDeg: Float,
    val panoramaXFraction: Float,
    val panoramaWidthFraction: Float,
    val keyframeIndex: Int,
    /** Normalised vertical centre (0–1) in the stitched bitmap. Defaults to 0.5 (middle). */
    val panoramaYFraction: Float = 0.5f,
    /** Normalised vertical span (0–1) of the bbox. Defaults to 1.0 (full height). */
    val panoramaHeightFraction: Float = 1.0f,
    val description: String = "",
    val color: Int = android.graphics.Color.TRANSPARENT,
    /**
     * Segmentation polygon in normalised panorama fractions [0,1].
     * Flat layout: [x0,y0, x1,y1, …].
     * Null when no segmentation data is available — renderer falls back to bbox rect.
     */
    val polygonXY: FloatArray? = null,
    val children: List<HierarchyNode> = emptyList(),
) {
    // FloatArray breaks data-class structural equality; we don't need it for nodes.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is HierarchyNode) return false
        return label == other.label && panoramaXFraction == other.panoramaXFraction
    }
    override fun hashCode(): Int = 31 * label.hashCode() + panoramaXFraction.hashCode()
}
