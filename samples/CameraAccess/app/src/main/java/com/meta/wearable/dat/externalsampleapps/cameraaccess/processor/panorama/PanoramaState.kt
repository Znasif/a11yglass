package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.Bitmap
import kotlin.math.abs

// ── Thresholds ──────────────────────────────────────────────────────────────
// At ~0.17fps (5-9s per ONNX inference), a comfortable 5°/s pan accumulates
// 25-45° between measurements. MIN/MAX are calibrated accordingly.
const val MIN_CAPTURE_DEGREES = 3f       // Below → too close, skip
const val MAX_CAPTURE_DEGREES = 50f      // Above → too fast, discard  (was 20°)
const val GAP_WARN_DEGREES    = 80f      // Gap marker threshold in strip (was 40°)
const val CAMERA_FOV_DEGREES  = 65f      // Assumed horizontal FOV
const val PX_PER_DEG          = 4f       // Strip pixels per degree
const val STRIP_HEIGHT_FRACTION = 0.22f  // Bottom fraction used by strip
const val STRIP_VISIBLE_RANGE = 90f      // ±90° visible in strip at once

// ── Phase state machine ──────────────────────────────────────────────────────
enum class PanoramaPhase {
    IDLE,              // No active session
    CAPTURING,         // Sweep in progress (Phase 1)
    STITCHING,         // Stitching running synchronously inside process()
    HIERARCHY_BUILDING,// Hierarchy builder running in processorScope (unused as observable state for now)
    REALITY_PROXY,     // Interactive localization mode (Phase 3)
}

// ── Package-level geometry helpers ───────────────────────────────────────────
// Moved here so both PanoramaProcessor and PanoramaLocalizer can call them
// without duplicating the math.

/**
 * Extract horizontal pixel shift from a 3×3 row-major homography.
 *
 * H maps the *reference* frame → *current* frame.
 * Projects the image centre from reference space into current space; the
 * difference tells us how far (in pixels) the scene has shifted horizontally.
 * When the camera pans right the scene moves left → xCur < cx → shift > 0.
 */
fun extractHorizontalShift(H: FloatArray, frameWidth: Int, frameHeight: Int): Float {
    val cx = frameWidth  / 2f
    val cy = frameHeight / 2f
    val w  = H[6] * cx + H[7] * cy + H[8]
    if (w == 0f) return 0f
    val xCur = (H[0] * cx + H[1] * cy + H[2]) / w
    return cx - xCur
}

/**
 * Extract vertical pixel shift from a 3×3 row-major homography.
 *
 * Same projection as horizontal, but reads the Y coordinate.
 * When the camera tilts down the scene moves up → yCur < cy → shift > 0.
 */
fun extractVerticalShift(H: FloatArray, frameWidth: Int, frameHeight: Int): Float {
    val cx = frameWidth  / 2f
    val cy = frameHeight / 2f
    val w  = H[6] * cx + H[7] * cy + H[8]
    if (w == 0f) return 0f
    val yCur = (H[3] * cx + H[4] * cy + H[5]) / w
    return cy - yCur
}

/**
 * A single accepted frame in the panorama sweep.
 *
 * @param bitmap            Full-resolution copy kept for Phase-2 stitching.
 * @param thumbnail         Strip-height scaled copy for live preview.
 * @param angleDeg          Accumulated horizontal angular position when this frame was captured.
 * @param verticalOffsetPx  Accumulated vertical pixel offset from the H chain.
 *                          Positive = canvas row shifted downward (camera tilted down).
 * @param homography        3×3 row-major H mapping the *previous* keyframe → this one.
 *                          The first keyframe carries an identity matrix.
 */
data class Keyframe(
    val bitmap: Bitmap,
    val thumbnail: Bitmap,
    val angleDeg: Float,
    val verticalOffsetPx: Float,
    val homography: FloatArray
) {
    // FloatArray doesn't play nicely with data-class equals/hashCode, but we
    // never use structural equality on Keyframes, so this is fine.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Keyframe) return false
        return angleDeg == other.angleDeg && bitmap === other.bitmap
    }

    override fun hashCode(): Int = System.identityHashCode(bitmap)
}

/**
 * Mutable state for an ongoing panorama capture session.
 * All mutations happen on a single coroutine (Dispatchers.Default), so no
 * additional synchronisation is needed beyond the @Volatile flags in the
 * processor itself.
 */
class PanoramaState {
    // Phase replaces the old isCapturing boolean; computed property kept for compat.
    var phase: PanoramaPhase = PanoramaPhase.IDLE
    val isCapturing: Boolean get() = phase == PanoramaPhase.CAPTURING

    var currentAngleDeg: Float = 0f
    var currentVerticalPx: Float = 0f  // Accumulated vertical pixel shift from H chain
    val keyframes: MutableList<Keyframe> = mutableListOf()
    var lastAcceptedFrame: Bitmap? = null  // Reference bitmap kept in sync with FeatureTracker
    var skippedCount: Int = 0
    var stitchedResult: Bitmap? = null     // Set when Phase 2 completes

    // Phase 3 state
    @Volatile var hierarchyNodes: List<HierarchyNode> = emptyList()
    var localizedAngleDeg: Float = 0f
    var focusedNode: HierarchyNode? = null

    fun reset() {
        phase = PanoramaPhase.IDLE
        currentAngleDeg = 0f
        currentVerticalPx = 0f
        keyframes.forEach {
            if (!it.bitmap.isRecycled)     it.bitmap.recycle()
            if (!it.thumbnail.isRecycled)  it.thumbnail.recycle()
        }
        keyframes.clear()
        lastAcceptedFrame?.let { if (!it.isRecycled) it.recycle() }
        lastAcceptedFrame = null
        skippedCount = 0
        // Do NOT recycle stitchedResult here — StreamUiState.processedFrame may still
        // hold a reference to the same bitmap, and Compose would crash drawing a recycled
        // bitmap. The UI clears processedFrame first (in capturePhoto), after which the
        // bitmap becomes unreachable and the GC reclaims it naturally.
        stitchedResult = null
        hierarchyNodes = emptyList()
        localizedAngleDeg = 0f
        focusedNode = null
    }
}
