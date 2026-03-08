package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.Bitmap
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens.FeatureTracker
import kotlin.math.abs

/**
 * Strip-based OR keyframe-based panorama localization for Reality Proxy.
 *
 * ## Keyframe mode (preferred — used after live sweeps)
 * Matches the live frame against the original captured keyframe bitmaps.
 * These are exact camera captures with no blending artifacts, giving far
 * better SuperPoint+LightGlue feature matching than stitched panorama strips.
 *
 *   extractHorizontalShift(H, ...) → shiftPx (how far live has panned right of keyframe)
 *   panorama_x = (kf.angleDeg − minAngle) × pxPerDeg + liveW/2 + shiftPx
 *   fraction   = panorama_x / panoramaWidth
 *
 * ## Strip mode (fallback — used for loaded panoramas without live keyframes)
 * Pre-crops the stitched panorama into overlapping strips, each the same
 * pixel width as the live camera frame. invertAndProject maps the live
 * frame centre into strip pixel space to get the panorama fraction.
 *
 * ## State machine (shared)
 * - **Locked** (lastBestIdx ≥ 0): re-test the same keyframe/strip.
 *   Hit → return updated fraction. Miss → unlock, restart scan.
 * - **Scanning** (lastBestIdx < 0): test [scanIdx], advance by 1 per call.
 *   Hit → lock. Miss → advance, return lastKnownFraction (sticky pointer).
 *
 * Threading: localize() is called from process() on Dispatchers.Default.
 */
class PanoramaLocalizer {

    // ── Strip-mode data ───────────────────────────────────────────────────────
    private data class Strip(val bitmap: Bitmap, val startX: Int, val panoramaWidth: Int)
    private val strips       = mutableListOf<Strip>()
    private var stripWidthPx = 0

    // ── Keyframe-mode data ────────────────────────────────────────────────────
    private data class KfRef(val bitmap: Bitmap, val angleDeg: Float)
    private val kfRefs      = mutableListOf<KfRef>()
    private var kfPanoramaW = 0
    private var kfMinAngle  = 0f
    private var kfPxPerDeg  = 0f
    private var kfFrameW    = 0

    // ── Mode flag ─────────────────────────────────────────────────────────────
    private var useKeyframes = false

    // ── Common scan/lock state ────────────────────────────────────────────────
    private var lastBestIdx       = -1
    private var scanIdx           = 0
    private var lastKnownFraction: Float? = null
    /** Avoids redundant setReferenceFrame calls when locked on the same index. */
    private var lastSetIdx        = -1

    companion object {
        private const val TAG                = "PanoramaLocalizer"
        private const val SEARCH_RADIUS      = 2
        /**
         * Both the reference (strip/keyframe) and the live frame are downsampled
         * by this factor before being passed to FeatureTracker, cutting inference
         * time roughly proportionally. Must be < 1f so kfRef/strip bitmaps are
         * always new copies (owned by this class and recycled in reset/reinit).
         */
        private const val LOCALIZATION_SCALE = 0.5f
    }

    /** Returns a new bitmap downsampled by [LOCALIZATION_SCALE]. Always a new allocation. */
    private fun downsample(src: Bitmap): Bitmap {
        val w = (src.width  * LOCALIZATION_SCALE).toInt().coerceAtLeast(1)
        val h = (src.height * LOCALIZATION_SCALE).toInt().coerceAtLeast(1)
        return Bitmap.createScaledBitmap(src, w, h, true)
    }

    // ── Public initialization ─────────────────────────────────────────────────

    /**
     * Keyframe-based initialization — preferred after a live sweep.
     *
     * The original captured keyframe bitmaps have no stitching artifacts, so
     * SuperPoint+LightGlue matches them against a live frame much more reliably
     * than stitched panorama crops.
     *
     * @param keyframes     Accepted keyframes sorted by angleDeg.
     * @param panoramaWidth Pixel width of the stitched panorama.
     * @param frameWidth    Live camera frame width (pixels).
     * @param frameHeight   Live camera frame height (pixels).
     */
    fun initializeWithKeyframes(
        keyframes: List<Keyframe>,
        panoramaWidth: Int,
        frameWidth: Int,
        frameHeight: Int,
    ) {
        strips.forEach { it.bitmap.recycle() }
        strips.clear()
        kfRefs.forEach { it.bitmap.recycle() }   // downsampled copies — always owned here
        kfRefs.clear()
        resetScanState()
        useKeyframes = true

        kfPanoramaW = panoramaWidth
        kfFrameW    = frameWidth
        kfMinAngle  = keyframes.minOfOrNull { it.angleDeg } ?: 0f
        kfPxPerDeg  = frameWidth.toFloat() / cameraHFovDeg(frameWidth, frameHeight)

        for (kf in keyframes) kfRefs.add(KfRef(downsample(kf.bitmap), kf.angleDeg))

        Log.d(TAG, "KF init: ${kfRefs.size} keyframes, " +
            "pxPerDeg=${"%.2f".format(kfPxPerDeg)} " +
            "minAngle=${"%.1f".format(kfMinAngle)}° " +
            "panoramaWidth=$panoramaWidth")
    }

    /**
     * Strip-based initialization — fallback when no live keyframes are available
     * (e.g. loading a saved panorama).
     *
     * Slices [panorama] into overlapping strips of [stripWidthPx] × [frameHeight] px.
     * Centre-crops the panorama vertically so the strip height matches the live frame.
     */
    fun initialize(panorama: Bitmap, stripWidthPx: Int, frameHeight: Int = panorama.height) {
        strips.forEach { it.bitmap.recycle() }
        strips.clear()
        kfRefs.forEach { it.bitmap.recycle() }   // downsampled copies — always owned here
        kfRefs.clear()
        resetScanState()
        useKeyframes = false

        this.stripWidthPx = stripWidthPx.coerceIn(1, panorama.width)

        val cropH   = frameHeight.coerceIn(1, panorama.height)
        val yOffset = (panorama.height - cropH) / 2
        val stride  = (this.stripWidthPx / 2).coerceAtLeast(1)

        var x = 0
        while (x + this.stripWidthPx <= panorama.width) {
            val crop = Bitmap.createBitmap(panorama, x, yOffset, this.stripWidthPx, cropH)
            strips.add(Strip(
                bitmap        = downsample(crop).also { if (it !== crop) crop.recycle() },
                startX        = x,
                panoramaWidth = panorama.width,
            ))
            x += stride
        }
        if (strips.isEmpty()) {
            val crop = Bitmap.createBitmap(panorama, 0, yOffset, panorama.width, cropH)
            strips.add(Strip(downsample(crop).also { if (it !== crop) crop.recycle() }, 0, panorama.width))
        }

        Log.d(TAG, "Strip init: ${strips.size} strips × ${this.stripWidthPx}×${cropH}px " +
            "from ${panorama.width}×${panorama.height}")
    }

    // ── Localization dispatch ─────────────────────────────────────────────────

    /**
     * Test ONE keyframe/strip against [liveFrame] and return a 0–1 X fraction.
     *
     * Returns the last known fraction (sticky pointer) while scanning.
     * Returns null only before any successful localization.
     */
    fun localize(liveFrame: Bitmap, featureTracker: FeatureTracker): Float? =
        if (useKeyframes) localizeWithKeyframes(liveFrame, featureTracker)
        else              localizeWithStrips(liveFrame, featureTracker)

    // ── Keyframe-based localization ───────────────────────────────────────────

    private fun localizeWithKeyframes(liveFrame: Bitmap, featureTracker: FeatureTracker): Float? {
        if (kfRefs.isEmpty()) return null

        val idx  = if (lastBestIdx >= 0) lastBestIdx else scanIdx
        val kf   = kfRefs[idx]
        val mode = if (lastBestIdx >= 0) "LOCKED" else "SCAN[$scanIdx/${kfRefs.size}]"

        // Skip redundant setReferenceFrame when locked on the same keyframe.
        if (idx != lastSetIdx) {
            featureTracker.setReferenceFrame(kf.bitmap, emptyList())  // kf.bitmap is already downsampled
            lastSetIdx = idx
        }

        val scaledLive = downsample(liveFrame)
        val scaledW    = scaledLive.width
        val scaledH    = scaledLive.height
        val t0         = System.currentTimeMillis()
        val H          = featureTracker.computeHomography(scaledLive)
        val matchMs    = System.currentTimeMillis() - t0
        scaledLive.recycle()

        if (H != null) {
            // shiftPx > 0 → live camera has panned RIGHT relative to this keyframe.
            // H is in downsampled pixel space; scale back to original for all downstream math.
            val shiftPx  = extractHorizontalShift(H, scaledW, scaledH) / LOCALIZATION_SCALE
            val shiftDeg = if (kfPxPerDeg > 0f) shiftPx / kfPxPerDeg else 0f

            // Accept the match only if the live frame meaningfully overlaps the keyframe.
            // |shiftPx| ≤ frameWidth means at least some content is shared.
            if (abs(shiftPx) <= kfFrameW.toFloat()) {
                lastBestIdx = idx

                // Panorama x of the live frame centre:
                //   keyframe left-edge  = (angleDeg − minAngle) × pxPerDeg
                //   + half live width   = centre of keyframe span
                //   + shiftPx           = additional rightward offset
                val panoramaX = (kf.angleDeg - kfMinAngle) * kfPxPerDeg +
                    liveFrame.width / 2f + shiftPx
                val fraction  = (panoramaX / kfPanoramaW).coerceIn(0f, 1f)
                val estAngle  = kf.angleDeg + shiftDeg
                lastKnownFraction = fraction

                Log.d(TAG, "[$mode] KF $idx/${kfRefs.size} kf=${kf.angleDeg}° " +
                    "shift=${"%.1f".format(shiftPx)}px (${"%.1f".format(shiftDeg)}°) " +
                    "→ est=${"%.1f".format(estAngle)}° frac=${"%.3f".format(fraction)} " +
                    "(${matchMs}ms)")
                return fraction
            } else {
                Log.d(TAG, "[$mode] KF $idx/${kfRefs.size} kf=${kf.angleDeg}° " +
                    "shift=${"%.1f".format(shiftPx)}px EXCEEDS frame width $kfFrameW — rejected " +
                    "(${matchMs}ms)")
            }
        } else {
            Log.d(TAG, "[$mode] KF $idx/${kfRefs.size} kf=${kf.angleDeg}° " +
                "H=null (no feature match) (${matchMs}ms)")
        }

        // Miss: advance the scan.
        if (lastBestIdx >= 0) {
            scanIdx     = (lastBestIdx - SEARCH_RADIUS).coerceAtLeast(0)
            lastBestIdx = -1
            lastSetIdx  = -1
            Log.d(TAG, "KF lock lost on idx=$idx (${kf.angleDeg}°) — rescanning from $scanIdx")
        } else {
            scanIdx = (scanIdx + 1) % kfRefs.size
            Log.v(TAG, "KF scan advance → $scanIdx/${kfRefs.size}")
        }

        // Return last known fraction so the pointer stays visible during scan.
        return lastKnownFraction
    }

    // ── Strip-based localization ──────────────────────────────────────────────

    private fun localizeWithStrips(liveFrame: Bitmap, featureTracker: FeatureTracker): Float? {
        if (strips.isEmpty()) return null

        val idx   = if (lastBestIdx >= 0) lastBestIdx else scanIdx
        val strip = strips[idx]
        val mode  = if (lastBestIdx >= 0) "LOCKED" else "SCAN[$scanIdx/${strips.size}]"

        if (idx != lastSetIdx) {
            featureTracker.setReferenceFrame(strip.bitmap, emptyList())  // strip.bitmap is already downsampled
            lastSetIdx = idx
        }
        val scaledLive = downsample(liveFrame)
        val scaledW    = scaledLive.width
        val scaledH    = scaledLive.height
        val t0         = System.currentTimeMillis()
        val H          = featureTracker.computeHomography(scaledLive)
        val matchMs    = System.currentTimeMillis() - t0
        scaledLive.recycle()

        // strip.bitmap.width is the downsampled strip width — the correct bounds for px.
        val scaledStripW = strip.bitmap.width.toFloat()

        if (H != null) {
            val px = invertAndProject(H, scaledW, scaledH)
            if (px != null && px >= 0f && px <= scaledStripW) {
                lastBestIdx = idx
                // Convert px from downsampled strip space back to original panorama coordinates.
                val pxOriginal = px / LOCALIZATION_SCALE
                val fraction   = (strip.startX + pxOriginal) / strip.panoramaWidth.toFloat()
                lastKnownFraction = fraction.coerceIn(0f, 1f)
                Log.d(TAG, "[$mode] Strip $idx/${strips.size} startX=${strip.startX} " +
                    "px=${"%.0f".format(pxOriginal)}/$stripWidthPx " +
                    "→ frac=${"%.3f".format(lastKnownFraction!!)} (${matchMs}ms)")
                return lastKnownFraction
            } else {
                val pxLog = px?.let { "%.0f".format(it / LOCALIZATION_SCALE) } ?: "null"
                Log.d(TAG, "[$mode] Strip $idx/${strips.size}: H ok " +
                    "px=$pxLog out-of-bounds [0..$stripWidthPx] — miss (${matchMs}ms)")
            }
        } else {
            Log.d(TAG, "[$mode] Strip $idx/${strips.size}: H=null (no feature match) — miss (${matchMs}ms)")
        }

        if (lastBestIdx >= 0) {
            scanIdx     = (lastBestIdx - SEARCH_RADIUS).coerceAtLeast(0)
            lastBestIdx = -1
            lastSetIdx  = -1
            Log.d(TAG, "Strip lock lost on idx=$idx — rescanning from $scanIdx")
        } else {
            scanIdx = (scanIdx + 1) % strips.size
            Log.v(TAG, "Strip scan advance → $scanIdx/${strips.size}")
        }

        return lastKnownFraction
    }

    /**
     * Invert H (strip→live) and project the live frame centre into strip pixel space.
     * Returns the X coordinate within the strip, or null if H is degenerate.
     */
    private fun invertAndProject(H: FloatArray, liveW: Int, liveH: Int): Float? {
        val det = H[0] * (H[4] * H[8] - H[5] * H[7]) -
                  H[1] * (H[3] * H[8] - H[5] * H[6]) +
                  H[2] * (H[3] * H[7] - H[4] * H[6])
        if (det == 0f) return null

        val i0 = (H[4] * H[8] - H[5] * H[7]) / det
        val i1 = (H[2] * H[7] - H[1] * H[8]) / det
        val i2 = (H[1] * H[5] - H[2] * H[4]) / det
        val i6 = (H[3] * H[7] - H[4] * H[6]) / det
        val i7 = (H[1] * H[6] - H[0] * H[7]) / det
        val i8 = (H[0] * H[4] - H[1] * H[3]) / det

        val lx = liveW / 2f
        val ly = liveH / 2f
        val w  = i6 * lx + i7 * ly + i8
        if (w == 0f) return null
        return (i0 * lx + i1 * ly + i2) / w
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    fun reset() {
        strips.forEach { it.bitmap.recycle() }
        strips.clear()
        kfRefs.forEach { it.bitmap.recycle() }   // downsampled copies — always owned here
        kfRefs.clear()
        stripWidthPx = 0
        kfPanoramaW  = 0; kfFrameW = 0; kfMinAngle = 0f; kfPxPerDeg = 0f
        resetScanState()
        useKeyframes = false
    }

    private fun resetScanState() {
        lastBestIdx       = -1
        scanIdx           = 0
        lastKnownFraction = null
        lastSetIdx        = -1
    }
}
