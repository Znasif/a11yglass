package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens.FeatureTracker
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * PanoramaProcessor — vision-only panorama capture using SuperPoint+LightGlue.
 *
 * id = -108
 *
 * ## Two-phase architecture
 *
 * **Phase 1 – Collection (live, during sweep)**
 * Every incoming frame is compared against the last accepted keyframe via
 * FeatureTracker. Frames are classified as:
 *   - *too-close* (|dx| < MIN_CAPTURE_DEGREES) → skipped, redundant
 *   - *too-fast*  (|dx| > MAX_CAPTURE_DEGREES) → discarded, reference reset
 *   - *accepted*  (MIN ≤ |dx| ≤ MAX)           → stored as a new Keyframe
 *
 * The live strip preview shows thumbnails at their angular positions.
 *
 * **Phase 2 – Stitching (post-hoc, after user stops)**
 * All stored keyframes + pairwise H matrices are chained into a single canvas.
 * The stitched result is shown in the strip until the next session.
 *
 * ## Threading
 * - `process()` runs on Dispatchers.Default (single writer, no locks needed).
 * - `startPanorama()` / `stopPanorama()` are called from the UI thread and
 *   write @Volatile flags that are checked at the start of each `process()` call.
 * - `AtomicBoolean isProcessing` drops concurrent frames to avoid queuing.
 */
class PanoramaProcessor : OnDeviceProcessor {

    companion object {
        private const val TAG = "PanoramaProcessor"
        private val IDENTITY_H = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

        // ── Guidance constants ────────────────────────────────────────────────
        private const val MILESTONE_DEG   = 20f    // announce at 30°, 60°, 90° …
        private const val SLOW_THRESHOLD  = 3      // consecutive non-advancing frames → nudge
        private const val MIN_GUIDANCE_MS = 8_000L // minimum gap between any two TTS lines
    }

    override val id   = -108
    override val name = "Panorama (On-Device)"
    override val description = "Sweep horizontally to capture a panorama"

    // ── Sub-components ───────────────────────────────────────────────────────
    private var featureTracker: FeatureTracker? = null
    private val state        = PanoramaState()
    private val stitcher     = PanoramaStitcher()
    private val stripRenderer = StripRenderer()

    // ── Concurrency guards ───────────────────────────────────────────────────
    private val isProcessing = AtomicBoolean(false)
    @Volatile private var startRequested = false
    @Volatile private var stopRequested  = false

    // ── Guidance state (reset each session) ──────────────────────────────────
    private var lastSpokenMilestoneDeg = 0f   // last milestone already announced
    private var consecutiveStillFrames = 0    // frames where camera didn't advance
    private var lastGuidanceMs         = 0L   // wall-clock time of last TTS utterance

    // ── Public state ─────────────────────────────────────────────────────────
    val isCapturing: Boolean get() = state.isCapturing

    // ── Overlay paint ────────────────────────────────────────────────────────
    private val statusPaint = Paint().apply {
        color = Color.WHITE
        textSize = 42f
        isAntiAlias = true
        isFakeBoldText = true
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }

    // ── OnDeviceProcessor ────────────────────────────────────────────────────

    override fun initialize(context: Context) {
        featureTracker = FeatureTracker(context)
        Log.d(TAG, "PanoramaProcessor initialized")
    }

    override suspend fun process(frame: Bitmap): OnDeviceProcessorResult =
        withContext(Dispatchers.Default) {
            val t0 = System.currentTimeMillis()

            // Frame-drop guard: if a previous frame is still being processed,
            // return the raw frame immediately so the UI stays responsive.
            if (!isProcessing.compareAndSet(false, true)) {
                return@withContext OnDeviceProcessorResult(
                    processedImage = frame,
                    text = null,
                    processingTimeMs = 0
                )
            }

            try {
                // ── Handle start request ─────────────────────────────────────
                if (startRequested) {
                    startRequested = false
                    state.reset()
                    state.isCapturing = true
                    lastSpokenMilestoneDeg = 0f
                    consecutiveStillFrames = 0
                    lastGuidanceMs = 0L

                    featureTracker?.setReferenceFrame(frame, emptyList())
                    state.lastAcceptedFrame = frame.copy(Bitmap.Config.ARGB_8888, false)
                    captureKeyframe(frame, 0f, IDENTITY_H.copyOf())
                    Log.d(TAG, "Panorama session started — first tile at 0°")

                    return@withContext OnDeviceProcessorResult(
                        processedImage = renderFrame(frame),
                        text = "Panorama session started",
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── Handle stop request ──────────────────────────────────────
                if (stopRequested) {
                    stopRequested = false
                    state.isCapturing = false

                    if (state.keyframes.size >= 2) {
                        Log.d(TAG, "Phase 2: stitching ${state.keyframes.size} keyframes…")
                        state.stitchedResult = stitcher.stitch(state.keyframes)
                    } else {
                        Log.d(TAG, "Not enough keyframes to stitch (${state.keyframes.size})")
                    }

                    val msg = "Panorama complete, ${state.keyframes.size} frames stitched"
                    Log.d(TAG, msg)
                    // Return the stitched panorama full-screen, not wrapped in the live strip overlay.
                    // Subsequent process() calls will keep returning it (see below) until a new
                    // session starts, so the result stays frozen in the UI.
                    return@withContext OnDeviceProcessorResult(
                        processedImage = state.stitchedResult ?: renderFrame(frame),
                        text = msg,
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── If idle with a completed stitch, keep returning it ────────
                // This freezes the processedFrame on the result between the stop
                // and the moment StreamViewModel cancels the localProcessingJob.
                if (!state.isCapturing && state.stitchedResult != null) {
                    return@withContext OnDeviceProcessorResult(
                        processedImage = state.stitchedResult,
                        text = null,
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── Normal capture loop ──────────────────────────────────────
                var guidanceText: String? = null

                if (state.isCapturing) {
                    val now      = System.currentTimeMillis()
                    val canSpeak = now - lastGuidanceMs > MIN_GUIDANCE_MS
                    val H        = featureTracker?.computeHomography(frame)

                    if (H == null) {
                        // Bad frame: motion blur, texture-less area, no reference set yet
                        state.skippedCount++
                        consecutiveStillFrames++
                        Log.d(TAG, "Bad frame (null H) — skipping [total skipped=${state.skippedCount}]")
                        if (canSpeak && consecutiveStillFrames >= SLOW_THRESHOLD) {
                            consecutiveStillFrames = 0
                            lastGuidanceMs = now
                            guidanceText = "Keep moving"
                        }
                    } else {
                        val shiftPx  = extractHorizontalShift(H, frame.width, frame.height)
                        val shiftDeg = shiftPx / frame.width * CAMERA_FOV_DEGREES

                        when {
                            abs(shiftDeg) < MIN_CAPTURE_DEGREES -> {
                                state.skippedCount++
                                consecutiveStillFrames++
                                Log.v(TAG, "Too close (${"%.1f".format(shiftDeg)}°) — skip")
                                if (canSpeak && consecutiveStillFrames >= SLOW_THRESHOLD) {
                                    consecutiveStillFrames = 0
                                    lastGuidanceMs = now
                                    guidanceText = "Keep moving"
                                }
                            }

                            abs(shiftDeg) > MAX_CAPTURE_DEGREES -> {
                                consecutiveStillFrames = 0
                                Log.d(TAG, "Too fast (${"%.1f".format(shiftDeg)}°) — discarded, resetting reference")
                                // Re-anchor to current frame so the next comparison is local
                                featureTracker?.setReferenceFrame(frame, emptyList())
                                state.lastAcceptedFrame?.let { if (!it.isRecycled) it.recycle() }
                                state.lastAcceptedFrame = frame.copy(Bitmap.Config.ARGB_8888, false)
                            }

                            else -> {
                                consecutiveStillFrames = 0
                                state.currentAngleDeg += shiftDeg
                                captureKeyframe(frame, state.currentAngleDeg, H)
                                featureTracker?.setReferenceFrame(frame, emptyList())
                                state.lastAcceptedFrame?.let { if (!it.isRecycled) it.recycle() }
                                state.lastAcceptedFrame = frame.copy(Bitmap.Config.ARGB_8888, false)
                                // Announce angle milestone (30°, 60°, 90° …)
                                val milestone = (state.currentAngleDeg / MILESTONE_DEG).toInt() * MILESTONE_DEG
                                if (canSpeak && milestone >= MILESTONE_DEG && milestone > lastSpokenMilestoneDeg) {
                                    lastSpokenMilestoneDeg = milestone
                                    lastGuidanceMs = now
                                    guidanceText = "${milestone.toInt()} degrees"
                                }
                            }
                        }
                    }
                }

                // ── Render strip overlay ─────────────────────────────────────
                OnDeviceProcessorResult(
                    processedImage = renderFrame(frame),
                    text = guidanceText,
                    processingTimeMs = System.currentTimeMillis() - t0
                )

            } finally {
                isProcessing.set(false)
            }
        }

    override fun release() {
        featureTracker?.release()
        featureTracker = null
        state.reset()
        stitcher.release()
        Log.d(TAG, "PanoramaProcessor released")
    }

    // ── Public session control ───────────────────────────────────────────────

    /** Start a new panorama sweep (UI thread safe). */
    fun startPanorama() {
        stopRequested  = false
        startRequested = true
        Log.d(TAG, "startPanorama() requested")
    }

    /** Stop the current sweep and trigger Phase-2 stitching (UI thread safe). */
    fun stopPanorama() {
        startRequested = false
        stopRequested  = true
        Log.d(TAG, "stopPanorama() requested")
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /**
     * Create and store a keyframe for [frame] at [angleDeg].
     * The thumbnail is scaled to strip height so it renders quickly.
     */
    private fun captureKeyframe(frame: Bitmap, angleDeg: Float, H: FloatArray) {
        val thumbH = (frame.height * STRIP_HEIGHT_FRACTION).toInt().coerceAtLeast(1)
        val thumbW = (frame.width.toFloat() / frame.height * thumbH).toInt().coerceAtLeast(1)
        val thumbnail = Bitmap.createScaledBitmap(frame, thumbW, thumbH, true)
        val fullCopy  = frame.copy(Bitmap.Config.ARGB_8888, false)

        state.keyframes.add(Keyframe(fullCopy, thumbnail, angleDeg, H))
        state.keyframes.sortBy { it.angleDeg }
        Log.d(TAG, "Keyframe accepted: ${"%.1f".format(angleDeg)}°, total=${state.keyframes.size}")
    }

    /**
     * Extract horizontal pixel shift from a 3×3 row-major homography.
     *
     * H maps the *reference* frame → *current* frame.
     * Projects the image centre from reference space into current space; the
     * difference tells us how far (in pixels) the scene has shifted horizontally.
     * When the camera pans right the scene moves left → xCur < cx → shift > 0.
     */
    private fun extractHorizontalShift(H: FloatArray, frameWidth: Int, frameHeight: Int): Float {
        val cx = frameWidth  / 2f
        val cy = frameHeight / 2f
        val w  = H[6] * cx + H[7] * cy + H[8]
        if (w == 0f) return 0f
        val xCur = (H[0] * cx + H[1] * cy + H[2]) / w
        return cx - xCur
    }

    /**
     * Composite the strip preview onto a mutable copy of [frame] and return it.
     */
    private fun renderFrame(frame: Bitmap): Bitmap {
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        val statusText = when {
            state.stitchedResult != null ->
                "Panorama done — ${state.keyframes.size} frames stitched"
            state.isCapturing ->
                "${state.keyframes.size} frames | ${"%.0f".format(state.currentAngleDeg)}° | sweeping…"
            state.keyframes.isNotEmpty() ->
                "Stopped: ${state.keyframes.size} frames"
            else ->
                "Tap camera button to start panorama"
        }
        canvas.drawText(statusText, 20f, 60f, statusPaint)

        stripRenderer.draw(
            canvas,
            frame.width,
            frame.height,
            state.currentAngleDeg,
            state.keyframes.toList(),   // snapshot to avoid ConcurrentModificationException
            state.isCapturing,
            state.stitchedResult
        )

        return output
    }
}
