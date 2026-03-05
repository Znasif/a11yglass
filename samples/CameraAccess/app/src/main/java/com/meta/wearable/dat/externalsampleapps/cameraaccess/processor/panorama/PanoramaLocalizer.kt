package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.graphics.Bitmap
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens.FeatureTracker

/**
 * Strip-based panorama localization for Reality Proxy.
 *
 * Pre-crops the stitched panorama into overlapping strips, each the same angular width
 * as the live camera FOV. Matching the live frame against a strip removes the scale
 * mismatch from direct full-panorama registration.
 *
 * ## One inference per frame
 *
 * computeHomography is expensive (~3s on device). Each localize() call runs exactly
 * ONE inference:
 *
 * - **Locked** (lastBestIdx ≥ 0): re-test the current best strip. On hit → return
 *   updated fraction. On miss → unlock and set scanIdx near the last known position.
 * - **Scanning** (lastBestIdx < 0): test strip[scanIdx], advance cursor by 1.
 *   On hit → lock. On miss → advance, return lastKnownFraction (sticky pointer).
 *
 * This gives 1 Hz localization updates when inference takes 1s, regardless of how
 * many strips exist.
 *
 * Threading: localize() is called from process() on Dispatchers.Default.
 * initialize() / reset() are called from the same dispatcher context.
 */
class PanoramaLocalizer {

    private data class Strip(val bitmap: Bitmap, val startX: Int, val panoramaWidth: Int)

    private val strips           = mutableListOf<Strip>()
    private var stripWidthPx     = 0
    private var lastBestIdx      = -1
    private var scanIdx          = 0
    private var lastKnownFraction: Float? = null

    companion object {
        private const val TAG           = "PanoramaLocalizer"
        private const val SEARCH_RADIUS = 2  // where to restart scan after lock loss
    }

    /**
     * Slice [panorama] into overlapping strips of [stripWidthPx] width (stride = stripW/2).
     * Call once on entering REALITY_PROXY.
     */
    fun initialize(panorama: Bitmap, stripWidthPx: Int) {
        strips.forEach { it.bitmap.recycle() }
        strips.clear()
        lastBestIdx       = -1
        scanIdx           = 0
        lastKnownFraction = null
        this.stripWidthPx = stripWidthPx.coerceIn(1, panorama.width)

        val stride = (this.stripWidthPx / 2).coerceAtLeast(1)
        var x = 0
        while (x + this.stripWidthPx <= panorama.width) {
            strips.add(Strip(
                bitmap        = Bitmap.createBitmap(panorama, x, 0, this.stripWidthPx, panorama.height),
                startX        = x,
                panoramaWidth = panorama.width
            ))
            x += stride
        }
        if (strips.isEmpty()) {
            strips.add(Strip(panorama.copy(Bitmap.Config.ARGB_8888, false), 0, panorama.width))
        }
        Log.d(TAG, "Initialized: ${strips.size} strips × ${this.stripWidthPx}px " +
            "from ${panorama.width}×${panorama.height} panorama")
    }

    /**
     * Test ONE strip against [liveFrame] and return a 0–1 X fraction when matched.
     *
     * While locked on a strip: returns updated fraction every frame (fast path).
     * While scanning: advances one strip per call; returns last known fraction so
     * the pointer stays visible during the scan.
     */
    fun localize(liveFrame: Bitmap, featureTracker: FeatureTracker): Float? {
        if (strips.isEmpty()) return null

        val idx = if (lastBestIdx >= 0) lastBestIdx else scanIdx
        val strip = strips[idx]

        featureTracker.setReferenceFrame(strip.bitmap, emptyList())
        val H = featureTracker.computeHomography(liveFrame)

        if (H != null) {
            val px = invertAndProject(H, liveFrame.width, liveFrame.height)
            if (px != null && px >= 0f && px <= stripWidthPx) {
                lastBestIdx = idx
                val fraction = (strip.startX + px) / strip.panoramaWidth.toFloat()
                lastKnownFraction = fraction.coerceIn(0f, 1f)
                Log.d(TAG, "Localized: strip $idx px=${"%.0f".format(px)} → fraction=${"%.3f".format(lastKnownFraction!!)}")
                return lastKnownFraction
            }
        }

        // Miss: advance the scan
        if (lastBestIdx >= 0) {
            // Locked strip failed — restart scan near last known position
            scanIdx = (lastBestIdx - SEARCH_RADIUS).coerceAtLeast(0)
            lastBestIdx = -1
            Log.d(TAG, "Lock lost on strip $idx — rescanning from $scanIdx")
        } else {
            scanIdx = (scanIdx + 1) % strips.size
        }

        // Return last known fraction so the pointer stays visible during scan
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

    fun reset() {
        strips.forEach { it.bitmap.recycle() }
        strips.clear()
        stripWidthPx      = 0
        lastBestIdx       = -1
        scanIdx           = 0
        lastKnownFraction = null
    }
}
