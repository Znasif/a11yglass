package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.florence.FlorenceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens.FeatureTracker
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.abs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * PanoramaProcessor — three-phase panorama capture and Reality Proxy.
 *
 * id = -108
 *
 * ## Phase 1 – Collection (live, during sweep)
 * Every incoming frame is compared against the last accepted keyframe via
 * FeatureTracker. Frames are classified as:
 *   - *too-close* (|dx| < MIN_CAPTURE_DEGREES) → skipped, redundant
 *   - *too-fast*  (|dx| > MAX_CAPTURE_DEGREES) → discarded, reference reset
 *   - *accepted*  (MIN ≤ |dx| ≤ MAX)           → stored as a new Keyframe
 *
 * ## Phase 2 – Stitching (post-hoc, after user stops)
 * All stored keyframes are composited into a single panorama bitmap.
 * Hierarchy building begins immediately afterward in processorScope.
 *
 * ## Phase 3 – Reality Proxy (interactive, user re-enters)
 * Live center-square reticle tracks position in the panorama via HLOC-style
 * localization (FeatureTracker reused with stored keyframe as reference).
 * Pre-derived hierarchy nodes are overlaid; focused node is highlighted.
 *
 * ## Threading
 * - `process()` runs on Dispatchers.Default (single writer, no locks needed).
 * - Start/stop/proxy request flags are @Volatile and checked at the top of
 *   each `process()` call.
 * - `AtomicBoolean isProcessing` drops concurrent frames to avoid queuing.
 * - `processorScope` runs hierarchy building independently of the localProcessingJob.
 */
class PanoramaProcessor : OnDeviceProcessor {

    companion object {
        private const val TAG = "PanoramaProcessor"
        private val IDENTITY_H = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

        // ── Guidance constants ────────────────────────────────────────────────
        private const val MILESTONE_DEG   = 20f    // announce at 20°, 40°, 60° …
        private const val SLOW_THRESHOLD  = 3      // consecutive non-advancing frames → nudge
        private const val MIN_GUIDANCE_MS = 8_000L // minimum gap between any two TTS lines
    }

    override val id   = -108
    override val name = "Panorama (On-Device)"
    override val description = "Sweep horizontally to capture a panorama"

    // ── Sub-components ───────────────────────────────────────────────────────
    private var featureTracker: FeatureTracker? = null
    private val state              = PanoramaState()
    private val stitcher           = PanoramaStitcher()
    private val stripRenderer      = StripRenderer()
    private val hierarchyBuilder   = PanoramaHierarchyBuilder()
    private var localizer: PanoramaLocalizer? = null
    private val realityProxyRenderer = RealityProxyRenderer()

    /**
     * Optional Florence-2 processor used to build a semantic hierarchy after
     * stitching. Set via [setFlorenceProcessor] from [OnDeviceProcessorManager]
     * after both processors are created. If null, falls back to angle-labelled nodes.
     */
    private var florenceProcessor: FlorenceProcessor? = null

    fun setFlorenceProcessor(fp: FlorenceProcessor) {
        florenceProcessor = fp
    }

    // Long-lived scope for hierarchy building — survives localProcessingJob cancellation.
    private val processorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    // ── Concurrency guards ───────────────────────────────────────────────────
    private val isProcessing = AtomicBoolean(false)
    @Volatile private var startRequested         = false
    @Volatile private var stopRequested          = false
    @Volatile private var realityProxyRequested  = false
    @Volatile private var exitProxyRequested     = false

    // ── Guidance state (reset each session) ──────────────────────────────────
    private var lastSpokenMilestoneDeg = 0f
    private var consecutiveStillFrames = 0
    private var lastGuidanceMs         = 0L

    // ── Reality Proxy tracking ────────────────────────────────────────────────
    // Only announce when the focused node changes, to avoid per-frame TTS spam.
    private var lastAnnouncedNodeLabel: String? = null

    // ── Public state ─────────────────────────────────────────────────────────
    val isCapturing: Boolean      get() = state.isCapturing
    val phase: PanoramaPhase      get() = state.phase
    val hasStitchedResult: Boolean get() = state.stitchedResult != null

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
        localizer      = PanoramaLocalizer()
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
                // ── ① Handle start request ───────────────────────────────────
                if (startRequested) {
                    startRequested = false
                    state.reset()
                    state.phase = PanoramaPhase.CAPTURING
                    lastSpokenMilestoneDeg = 0f
                    consecutiveStillFrames = 0
                    lastGuidanceMs = 0L
                    lastAnnouncedNodeLabel = null

                    featureTracker?.setReferenceFrame(frame, emptyList())
                    state.lastAcceptedFrame = frame.copy(Bitmap.Config.ARGB_8888, false)
                    captureKeyframe(frame, 0f, 0f, IDENTITY_H.copyOf())
                    Log.d(TAG, "Panorama session started — first tile at 0°")

                    return@withContext OnDeviceProcessorResult(
                        processedImage = renderFrame(frame),
                        text = "Panorama session started",
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── ② Handle stop request ────────────────────────────────────
                if (stopRequested) {
                    stopRequested = false
                    state.phase = PanoramaPhase.STITCHING

                    if (state.keyframes.size >= 2) {
                        Log.d(TAG, "Phase 2: stitching ${state.keyframes.size} keyframes…")
                        state.stitchedResult = stitcher.stitch(state.keyframes)
                    } else {
                        Log.d(TAG, "Not enough keyframes to stitch (${state.keyframes.size})")
                    }

                    state.phase = PanoramaPhase.IDLE

                    // Launch scene analysis in processorScope (outlives localProcessingJob).
                    // Reality Proxy is gated on this completing (enterRealityProxy checks
                    // state.hierarchyNodes.isNotEmpty()).
                    val panorama         = state.stitchedResult
                    val panoramaWidth    = panorama?.width
                    val keyframeSnapshot = state.keyframes.toList()
                    if (panoramaWidth != null && panorama != null) {
                        val fp = florenceProcessor
                        processorScope.launch {
                            val nodes: List<HierarchyNode>
                            if (fp != null) {
                                Log.d(TAG, "Launching Florence scene analysis on ${panorama.width}×${panorama.height} panorama…")
                                val regions = fp.analyzeRegions(panorama)
                                nodes = if (regions.isNotEmpty()) {
                                    val minAngle = keyframeSnapshot.minOf { it.angleDeg }
                                    val maxAngle = keyframeSnapshot.maxOf { it.angleDeg }
                                    val fov = keyframeSnapshot.firstOrNull()?.bitmap?.let {
                                        cameraHFovDeg(it.width, it.height)
                                    } ?: 65f
                                    hierarchyBuilder.buildFromFlorence(
                                        regions, panoramaWidth, minAngle, maxAngle, fov
                                    ).also { Log.d(TAG, "Florence hierarchy: ${it.size} nodes") }
                                } else {
                                    Log.d(TAG, "Florence returned no regions — falling back to angle labels")
                                    hierarchyBuilder.build(keyframeSnapshot, panoramaWidth)
                                }
                            } else {
                                nodes = hierarchyBuilder.build(keyframeSnapshot, panoramaWidth)
                            }
                            hierarchyBuilder.setNodes(nodes)
                            state.hierarchyNodes = nodes
                            Log.d(TAG, "Hierarchy ready: ${nodes.size} nodes")
                        }
                    }

                    val msg = "Panorama complete, ${state.keyframes.size} frames stitched"
                    Log.d(TAG, msg)
                    return@withContext OnDeviceProcessorResult(
                        processedImage = state.stitchedResult ?: renderFrame(frame),
                        text = msg,
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── ③ Handle enter Reality Proxy request ─────────────────────
                if (realityProxyRequested) {
                    realityProxyRequested = false
                    state.phase = PanoramaPhase.REALITY_PROXY
                    lastAnnouncedNodeLabel = null
                    localizer?.initialize(state.keyframes, featureTracker!!)
                    Log.d(TAG, "Entered Reality Proxy mode")

                    return@withContext OnDeviceProcessorResult(
                        processedImage = renderRealityProxy(frame),
                        text = "Reality Proxy — look at objects",
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── ④ Handle exit Reality Proxy request ──────────────────────
                if (exitProxyRequested) {
                    exitProxyRequested = false
                    state.phase = PanoramaPhase.IDLE
                    localizer?.reset()
                    lastAnnouncedNodeLabel = null
                    Log.d(TAG, "Exited Reality Proxy mode")

                    return@withContext OnDeviceProcessorResult(
                        processedImage = state.stitchedResult ?: frame,
                        text = null,
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── ⑤ Idle with completed stitch: keep returning frozen result ─
                // Excludes REALITY_PROXY (which has its own live loop below).
                if (!state.isCapturing
                    && state.stitchedResult != null
                    && state.phase != PanoramaPhase.REALITY_PROXY
                ) {
                    val statusText = if (state.hierarchyNodes.isEmpty()) "Building scene map…" else null
                    return@withContext OnDeviceProcessorResult(
                        processedImage = state.stitchedResult,
                        text = statusText,
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── ⑥ Reality Proxy live frame loop ──────────────────────────
                if (state.phase == PanoramaPhase.REALITY_PROXY && state.stitchedResult != null) {
                    val angle = localizer?.localize(frame, state.keyframes, featureTracker!!)
                        ?: state.localizedAngleDeg
                    state.localizedAngleDeg = angle

                    val prevFocused = state.focusedNode
                    state.focusedNode = hierarchyBuilder.findNodeAt(angle)

                    // Only announce when the focused node changes to avoid per-frame TTS spam
                    val newLabel = state.focusedNode?.label
                    val announcement = if (newLabel != lastAnnouncedNodeLabel) {
                        lastAnnouncedNodeLabel = newLabel
                        newLabel
                    } else {
                        null
                    }

                    return@withContext OnDeviceProcessorResult(
                        processedImage = renderRealityProxy(frame),
                        text = announcement,
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── Normal Phase 1 capture loop ───────────────────────────────
                var guidanceText: String? = null

                if (state.isCapturing) {
                    val now      = System.currentTimeMillis()
                    val canSpeak = now - lastGuidanceMs > MIN_GUIDANCE_MS
                    val H        = featureTracker?.computeHomography(frame)

                    if (H == null) {
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
                        val shiftDeg = shiftPx / frame.width * cameraHFovDeg(frame.width, frame.height)

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
                                featureTracker?.setReferenceFrame(frame, emptyList())
                                state.lastAcceptedFrame?.let { if (!it.isRecycled) it.recycle() }
                                state.lastAcceptedFrame = frame.copy(Bitmap.Config.ARGB_8888, false)
                            }

                            else -> {
                                consecutiveStillFrames = 0
                                state.currentAngleDeg += shiftDeg
                                state.currentVerticalPx += extractVerticalShift(H, frame.width, frame.height)
                                captureKeyframe(frame, state.currentAngleDeg, state.currentVerticalPx, H)
                                featureTracker?.setReferenceFrame(frame, emptyList())
                                state.lastAcceptedFrame?.let { if (!it.isRecycled) it.recycle() }
                                state.lastAcceptedFrame = frame.copy(Bitmap.Config.ARGB_8888, false)
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

                // ── Render strip overlay (Phase 1 live preview) ───────────────
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
        processorScope.cancel()
        featureTracker?.release()
        featureTracker = null
        localizer?.reset()
        hierarchyBuilder.clear()
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

    /**
     * Enter Reality Proxy mode.
     *
     * No-op if no stitched result exists, or if scene analysis (Florence or
     * fallback hierarchy build) has not yet completed. The UI shows
     * "Building scene map…" while waiting; the user can try again once that
     * message disappears.
     */
    fun enterRealityProxy() {
        when {
            state.stitchedResult == null ->
                Log.d(TAG, "enterRealityProxy() ignored — no stitched result yet")
            state.hierarchyNodes.isEmpty() ->
                Log.d(TAG, "enterRealityProxy() blocked — scene analysis still running")
            else -> {
                realityProxyRequested = true
                Log.d(TAG, "enterRealityProxy() requested")
            }
        }
    }

    /** Exit Reality Proxy mode and return to frozen panorama display. */
    fun exitRealityProxy() {
        exitProxyRequested = true
        Log.d(TAG, "exitRealityProxy() requested")
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /**
     * Create and store a keyframe for [frame] at [angleDeg].
     * The thumbnail is scaled to strip height so it renders quickly.
     */
    private fun captureKeyframe(frame: Bitmap, angleDeg: Float, verticalOffsetPx: Float, H: FloatArray) {
        val thumbH = (frame.height * STRIP_HEIGHT_FRACTION).toInt().coerceAtLeast(1)
        val thumbW = (frame.width.toFloat() / frame.height * thumbH).toInt().coerceAtLeast(1)
        val thumbnail = Bitmap.createScaledBitmap(frame, thumbW, thumbH, true)
        val fullCopy  = frame.copy(Bitmap.Config.ARGB_8888, false)

        state.keyframes.add(Keyframe(fullCopy, thumbnail, angleDeg, verticalOffsetPx, H))
        state.keyframes.sortBy { it.angleDeg }
        Log.d(TAG, "Keyframe accepted: ${"%.1f".format(angleDeg)}° " +
              "Y=${"%.1f".format(verticalOffsetPx)}px, total=${state.keyframes.size}")
    }

    /**
     * Render the Phase-3 Reality Proxy composite for [frame].
     * Falls back to the strip overlay if no stitched result is available.
     */
    private fun renderRealityProxy(frame: Bitmap): Bitmap {
        val panorama = state.stitchedResult ?: return renderFrame(frame)
        val keyframes = state.keyframes
        if (keyframes.isEmpty()) return renderFrame(frame)

        return realityProxyRenderer.render(
            frame         = frame,
            panorama      = panorama,
            nodes         = state.hierarchyNodes,
            currentAngleDeg = state.localizedAngleDeg,
            minAngleDeg   = keyframes.minOf { it.angleDeg },
            maxAngleDeg   = keyframes.maxOf { it.angleDeg },
            focusedNode   = state.focusedNode,
        )
    }

    /**
     * Composite the Phase-1 strip preview onto a mutable copy of [frame].
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
            state.keyframes.toList(),
            state.isCapturing,
            state.stitchedResult
        )

        return output
    }
}
