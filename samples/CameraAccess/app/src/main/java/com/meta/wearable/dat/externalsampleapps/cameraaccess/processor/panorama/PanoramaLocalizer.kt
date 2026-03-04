package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.Bitmap
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens.FeatureTracker
import kotlin.math.abs

/**
 * HLOC-style live-to-panorama localization using the shared FeatureTracker.
 *
 * The panorama sweep (Phase 1) stores keyframes at known angles. During
 * REALITY_PROXY (Phase 3), each live camera frame is matched against the
 * nearest stored keyframe so that we can estimate which part of the panorama
 * the user is currently looking at.
 *
 * Threading: localize() is called from process() on Dispatchers.Default.
 * initialize() / reset() are called from the same dispatcher context.
 */
class PanoramaLocalizer {

    private var currentKeyframeIdx = 0
    private var referenceAngleDeg  = 0f

    companion object {
        private const val TAG = "PanoramaLocalizer"
    }

    /**
     * Set the initial reference keyframe (call once on entering REALITY_PROXY).
     * Anchors to the midpoint keyframe as starting estimate so localization
     * starts from the centre of the panorama regardless of sweep direction.
     */
    fun initialize(keyframes: List<Keyframe>, featureTracker: FeatureTracker) {
        if (keyframes.isEmpty()) return
        currentKeyframeIdx = keyframes.size / 2
        referenceAngleDeg  = keyframes[currentKeyframeIdx].angleDeg
        featureTracker.setReferenceFrame(keyframes[currentKeyframeIdx].bitmap, emptyList())
        Log.d(TAG, "Initialized at keyframe $currentKeyframeIdx (${referenceAngleDeg}°)")
    }

    /**
     * Match [liveFrame] against the current reference keyframe and return the
     * estimated panorama angle (degrees) of the live frame centre.
     *
     * Returns null on tracker failure; the caller should hold the last known angle.
     *
     * Side-effect: if the estimated angle drifts more than 60 % of one FOV from
     * the reference, the nearest keyframe becomes the new reference so the tracker
     * stays well-conditioned for future frames.
     *
     * Sign note: task.md has a typo and shows a minus sign here.  The correct sign
     * is POSITIVE because:
     *   - H maps reference keyframe → live frame
     *   - camera panning right → scene shifts left in live frame → shiftPx > 0
     *   - rightward pan = increasing panorama angle
     * Therefore: estimatedAngle = referenceAngleDeg + shiftDeg
     */
    fun localize(
        liveFrame: Bitmap,
        keyframes: List<Keyframe>,
        featureTracker: FeatureTracker,
    ): Float? {
        if (keyframes.isEmpty()) return null

        val H = featureTracker.computeHomography(liveFrame) ?: return null

        val shiftPx    = extractHorizontalShift(H, liveFrame.width, liveFrame.height)
        val hFov = cameraHFovDeg(liveFrame.width, liveFrame.height)
        val shiftDeg   = shiftPx / liveFrame.width * hFov
        val estimatedAngle = referenceAngleDeg + shiftDeg

        // Re-anchor when we've drifted far enough that a closer keyframe exists
        if (abs(estimatedAngle - referenceAngleDeg) > hFov * 0.6f) {
            val newIdx = keyframes.indices
                .minByOrNull { abs(keyframes[it].angleDeg - estimatedAngle) }!!
            if (newIdx != currentKeyframeIdx) {
                currentKeyframeIdx = newIdx
                referenceAngleDeg  = keyframes[newIdx].angleDeg
                featureTracker.setReferenceFrame(keyframes[newIdx].bitmap, emptyList())
                Log.d(TAG, "Re-anchored to keyframe $newIdx (${referenceAngleDeg}°)")
            }
        }

        val minAngle = keyframes.minOf { it.angleDeg }
        val maxAngle = keyframes.maxOf { it.angleDeg }
        return estimatedAngle.coerceIn(minAngle, maxAngle)
    }

    fun reset() {
        currentKeyframeIdx = 0
        referenceAngleDeg  = 0f
    }
}
