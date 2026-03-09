package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.florence.FlorenceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.sam.SamProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens.FeatureTracker
import kotlin.math.abs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
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
class PanoramaProcessor : OnDeviceProcessor() {

    companion object {
        private const val TAG = "PanoramaProcessor"
        const val PROCESSOR_ID = -108
        private val IDENTITY_H = floatArrayOf(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f)

        // ── Capture mode toggle ────────────────────────────────────────────
        // true  = offline: pixel-diff capture, pass-through display, async post-hoc homography
        // false = online:  per-frame homography, live strip preview, synchronous stitching
        const val OFFLINE_CAPTURE = false

        // ── Pixel-diff capture gating (offline mode only) ─────────────────
        private const val DIFF_THUMB_SIZE         = 64
        private const val PIXEL_DIFF_THRESHOLD    = 10    // mean abs diff per channel (0-255)
        private const val MIN_CAPTURE_INTERVAL_MS = 200L  // hard cap: ≤5 captures/sec

        // ── Guidance constants ────────────────────────────────────────────────
        private const val MILESTONE_DEG   = 5f    // announce at 20°, 40°, 60° …
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

    /**
     * Optional Florence-2 processor used to build a semantic hierarchy after
     * stitching. Set via [setFlorenceProcessor] from [OnDeviceProcessorManager]
     * after both processors are created. If null, falls back to angle-labelled nodes.
     */
    private var florenceProcessor: FlorenceProcessor? = null
    private var samProcessor:     SamProcessor?      = null

    fun setFlorenceProcessor(fp: FlorenceProcessor) { florenceProcessor = fp }
    fun setSamProcessor(sp: SamProcessor)            { samProcessor     = sp }

    // ── Hierarchy-ready signal ────────────────────────────────────────────────
    // Emits true once Florence / fallback hierarchy completes after stitching.
    // StreamViewModel collects this to activate the Explore button.
    private val _hierarchyReady = MutableStateFlow(false)
    val hierarchyReady: StateFlow<Boolean> = _hierarchyReady.asStateFlow()

    // ── Progressive node enrichment ───────────────────────────────────────────
    // Emits the full node list each time a node's longDescription is filled in
    // by the background captioning loop. StreamViewModel collects this to
    // re-save the .glassio file after each enriched node.
    private val _hierarchyNodesFlow = MutableStateFlow<List<HierarchyNode>>(emptyList())
    val hierarchyNodesFlow: StateFlow<List<HierarchyNode>> = _hierarchyNodesFlow.asStateFlow()

    // ── Localised angle (Reality Proxy) ──────────────────────────────────────
    // Absolute x-fraction (0..1) across the panorama, updated each frame in
    // REALITY_PROXY. StreamViewModel collects this to pan the panorama image.
    private val _localizedXFraction = MutableStateFlow(0f)
    val localizedXFraction: StateFlow<Float> = _localizedXFraction.asStateFlow()

    // ── Concurrency guards ───────────────────────────────────────────────────
    @Volatile private var startRequested         = false
    @Volatile private var stopRequested          = false
    @Volatile private var realityProxyRequested  = false
    @Volatile private var exitProxyRequested     = false

    // ── Guidance state (reset each session) ──────────────────────────────────
    private var lastSpokenMilestoneDeg = 0f
    private var consecutiveStillFrames = 0
    private var lastGuidanceMs         = 0L
    private var lastCaptureTimeMs      = 0L

    // ── Post-hoc processing progress (async, read by process() during STITCHING) ─
    @Volatile private var postHocComplete    = false
    @Volatile private var postHocAngleDeg    = 0f
    private var lastReportedPostHocMilestone  = 0

    // ── Reality Proxy tracking ────────────────────────────────────────────────
    // Only announce when the focused node changes, to avoid per-frame TTS spam.
    private var lastAnnouncedNodeLabel: String? = null

    // ── Public state ─────────────────────────────────────────────────────────
    val isCapturing: Boolean        get() = state.isCapturing
    val phase: PanoramaPhase        get() = state.phase
    val hasStitchedResult: Boolean  get() = state.stitchedResult != null
    val stitchedResult: Bitmap?     get() = state.stitchedResult
    val hierarchyNodeCount: Int     get() = state.hierarchyNodes.size
    val hierarchyNodes: List<HierarchyNode> get() = state.hierarchyNodes
    /** Angular span (degrees) of the current stitched panorama. */
    val panoramaAngularSpanDeg: Float get() = state.panoramaAngularSpanDeg
    /** Snapshot of accepted keyframes — used by StreamViewModel to save the .glassio sidecar. */
    val keyframes: List<Keyframe>   get() = state.keyframes.toList()

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
            // return null so the last strip-annotated frame stays on screen.
            if (!isProcessing.compareAndSet(false, true)) {
                return@withContext OnDeviceProcessorResult(
                    processedImage = null,
                    text = null,
                    processingTimeMs = 0
                )
            }

            try {
                // ── ① Handle start request ───────────────────────────────────
                if (startRequested) {
                    startRequested = false
                    realityProxyRequested = false
                    exitProxyRequested    = false
                    _hierarchyReady.value = false
                    state.reset()
                    state.phase = PanoramaPhase.CAPTURING
                    lastSpokenMilestoneDeg = 0f
                    consecutiveStillFrames = 0
                    lastGuidanceMs = 0L
                    lastCaptureTimeMs = System.currentTimeMillis()
                    lastAnnouncedNodeLabel = null

                    if (OFFLINE_CAPTURE) {
                        // Offline: store first raw frame + pixel-diff reference
                        state.rawFrames.add(frame.copy(Bitmap.Config.ARGB_8888, false))
                        state.lastCapturedSmall = Bitmap.createScaledBitmap(
                            frame, DIFF_THUMB_SIZE, DIFF_THUMB_SIZE, true
                        )
                        Log.d(TAG, "Panorama session started (offline) — first frame stored")
                        return@withContext OnDeviceProcessorResult(
                            processedImage = frame,
                            text = "Panorama session started",
                            processingTimeMs = System.currentTimeMillis() - t0
                        )
                    } else {
                        // Online: set up FeatureTracker + first keyframe
                        val ft = featureTracker!!
                        ft.setReferenceFrame(frame, emptyList())
                        state.lastAcceptedFrame = frame.copy(Bitmap.Config.ARGB_8888, false)
                        captureKeyframe(frame, 0f, 0f, IDENTITY_H.copyOf())
                        Log.d(TAG, "Panorama session started (online) — first keyframe captured")
                        return@withContext OnDeviceProcessorResult(
                            processedImage = renderFrame(frame),
                            text = "Panorama session started",
                            processingTimeMs = System.currentTimeMillis() - t0
                        )
                    }
                }

                // ── ② Handle stop request ────────────────────────────────────
                if (stopRequested) {
                    stopRequested = false

                    if (OFFLINE_CAPTURE) {
                        // ── Offline: async post-hoc homography + stitch ─────
                        val rawCount = state.rawFrames.size
                        if (rawCount >= 2) {
                            state.phase = PanoramaPhase.STITCHING
                            postHocComplete = false
                            postHocAngleDeg = 0f
                            lastReportedPostHocMilestone = 0

                            processorScope.launch {
                                Log.d(TAG, "Phase 2 (offline): processing $rawCount raw frames post-hoc…")
                                processFramesPostHoc()

                                Log.d(TAG, "Phase 2 (offline): stitching ${state.keyframes.size} keyframes…")
                                state.stitchedResult = stitcher.stitch(state.keyframes)
                                if (state.stitchedResult != null) {
                                    val hFov = state.keyframes.firstOrNull()?.bitmap
                                        ?.let { cameraHFovDeg(it.width, it.height) } ?: 65f
                                    val frameW = state.keyframes.firstOrNull()?.bitmap?.width ?: 720
                                    state.panoramaAngularSpanDeg =
                                        ((state.stitchedResult!!.width.toFloat() * hFov / frameW)
                                            .coerceIn(10f, 350f))
                                }
                                postHocComplete = true
                                Log.d(TAG, "Phase 2 (offline) async complete: ${state.keyframes.size} keyframes, " +
                                    "stitch=${if (state.stitchedResult != null) "${state.stitchedResult!!.width}×${state.stitchedResult!!.height}" else "null"}")
                            }

                            return@withContext OnDeviceProcessorResult(
                                processedImage = frame,
                                text = "Processing $rawCount frames",
                                processingTimeMs = System.currentTimeMillis() - t0
                            )
                        } else {
                            Log.d(TAG, "Not enough raw frames to stitch ($rawCount)")
                            state.rawFrames.forEach { if (!it.isRecycled) it.recycle() }
                            state.rawFrames.clear()
                            state.phase = PanoramaPhase.IDLE
                            return@withContext OnDeviceProcessorResult(
                                processedImage = frame,
                                text = "Not enough frames",
                                processingTimeMs = System.currentTimeMillis() - t0
                            )
                        }
                    } else {
                        // ── Online: synchronous stitch (keyframes already built) ──
                        val kfCount = state.keyframes.size
                        if (kfCount >= 2) {
                            state.phase = PanoramaPhase.STITCHING
                            Log.d(TAG, "Phase 2 (online): stitching $kfCount keyframes synchronously…")
                            state.stitchedResult = stitcher.stitch(state.keyframes)
                            if (state.stitchedResult != null) {
                                val hFov = state.keyframes.firstOrNull()?.bitmap
                                    ?.let { cameraHFovDeg(it.width, it.height) } ?: 65f
                                val frameW = state.keyframes.firstOrNull()?.bitmap?.width ?: 720
                                state.panoramaAngularSpanDeg =
                                    ((state.stitchedResult!!.width.toFloat() * hFov / frameW)
                                        .coerceIn(10f, 350f))
                            }
                            state.phase = PanoramaPhase.IDLE

                            // Launch Florence analysis asynchronously
                            val panorama      = state.stitchedResult
                            val panoramaWidth = panorama?.width
                            val keyframeSnapshot = state.keyframes.toList()
                            if (panoramaWidth != null) {
                                launchHierarchyAnalysis(panorama, panoramaWidth, keyframeSnapshot)
                            }

                            val msg = "Panorama complete, $kfCount frames stitched"
                            Log.d(TAG, msg)
                            return@withContext OnDeviceProcessorResult(
                                processedImage = state.stitchedResult ?: frame,
                                text = msg,
                                processingTimeMs = System.currentTimeMillis() - t0
                            )
                        } else {
                            Log.d(TAG, "Not enough keyframes to stitch ($kfCount)")
                            state.phase = PanoramaPhase.IDLE
                            return@withContext OnDeviceProcessorResult(
                                processedImage = frame,
                                text = "Not enough frames",
                                processingTimeMs = System.currentTimeMillis() - t0
                            )
                        }
                    }
                }

                // ── ②b Ongoing async stitching (offline mode only) ───────────
                if (state.phase == PanoramaPhase.STITCHING && OFFLINE_CAPTURE) {
                    if (postHocComplete) {
                        postHocComplete = false
                        state.phase = PanoramaPhase.IDLE

                        val panorama      = state.stitchedResult
                        val panoramaWidth = panorama?.width
                        val keyframeSnapshot = state.keyframes.toList()
                        if (panoramaWidth != null) {
                            launchHierarchyAnalysis(panorama, panoramaWidth, keyframeSnapshot)
                        }

                        val msg = "Panorama complete, ${state.keyframes.size} frames stitched"
                        Log.d(TAG, msg)
                        return@withContext OnDeviceProcessorResult(
                            processedImage = state.stitchedResult ?: frame,
                            text = msg,
                            processingTimeMs = System.currentTimeMillis() - t0
                        )
                    }

                    // Still processing — report 20° milestones
                    val angle = kotlin.math.abs(postHocAngleDeg)
                    val milestone = (angle / 20f).toInt() * 20
                    val progress = if (milestone >= 20 && milestone > lastReportedPostHocMilestone) {
                        lastReportedPostHocMilestone = milestone
                        "$milestone degrees aligned"
                    } else null

                    return@withContext OnDeviceProcessorResult(
                        processedImage = frame,
                        text = progress,
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── ③ Handle enter Reality Proxy request ─────────────────────
                if (realityProxyRequested) {
                    realityProxyRequested = false
                    state.phase = PanoramaPhase.REALITY_PROXY
                    lastAnnouncedNodeLabel = null
                    state.stitchedResult?.let { panorama ->
                        // Prefer keyframe-based localization: original captured bitmaps have
                        // no stitching/blending artifacts, giving much better feature matches
                        // than strips sliced from the composite panorama.
                        // Fall back to strip-based only for loaded panoramas (no keyframes).
                        if (state.keyframes.isNotEmpty()) {
                            localizer?.initializeWithKeyframes(
                                state.keyframes.toList(),
                                panorama.width,
                                frame.width,
                                frame.height,
                            )
                        } else {
                            val stripWidthPx = frame.width.coerceAtMost(panorama.width)
                            localizer?.initialize(panorama, stripWidthPx, frame.height)
                        }
                    }
                    Log.d(TAG, "Entered Reality Proxy mode — ${state.hierarchyNodes.size} nodes, " +
                        "${state.keyframes.size} keyframes")

                    return@withContext OnDeviceProcessorResult(
                        processedImage = frame,
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
                // TTS progress updates are driven by StreamViewModel.startHierarchyWatcher().
                if (!state.isCapturing
                    && state.stitchedResult != null
                    && state.phase != PanoramaPhase.REALITY_PROXY
                ) {
                    return@withContext OnDeviceProcessorResult(
                        processedImage = state.stitchedResult,
                        text = null,
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── ⑥ Reality Proxy live frame loop ──────────────────────────
                if (state.phase == PanoramaPhase.REALITY_PROXY && state.stitchedResult != null) {
                    Log.v(TAG, "RP frame ${frame.width}×${frame.height}")

                    val xFraction = localizer?.localize(frame, featureTracker!!)
                    if (xFraction != null) {
                        state.localizedAngleDeg = run {
                            val panorama = state.stitchedResult!!
                            val minAngle = state.keyframes.minOfOrNull { it.angleDeg } ?: 0f
                            val maxAngle = state.keyframes.maxOfOrNull { it.angleDeg }
                                ?.coerceAtLeast(minAngle + panorama.width.toFloat() / PX_PER_DEG)
                                ?: (panorama.width.toFloat() / PX_PER_DEG)
                            minAngle + xFraction * (maxAngle - minAngle)
                        }
                        _localizedXFraction.value = xFraction
                    }

                    state.focusedNode = hierarchyBuilder.findNodeAt(state.localizedAngleDeg)

                    // Only announce when the focused node changes to avoid per-frame TTS spam
                    val newLabel = state.focusedNode?.label
                    val announcement = if (newLabel != lastAnnouncedNodeLabel) {
                        lastAnnouncedNodeLabel = newLabel
                        newLabel
                    } else {
                        null
                    }

                    // Carousel WebView owns display — return raw frame so UI preview is unused
                    return@withContext OnDeviceProcessorResult(
                        processedImage = frame,
                        text = announcement,
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                // ── Phase 1 capture loop ─────────────────────────────────────
                if (state.isCapturing) {
                    if (OFFLINE_CAPTURE) {
                        // Offline: pixel-diff gated raw frame storage, pass-through display
                        if (shouldCapture(frame)) {
                            state.rawFrames.add(frame.copy(Bitmap.Config.ARGB_8888, false))
                            Log.v(TAG, "Raw frame captured: ${state.rawFrames.size} total")
                        }
                        return@withContext OnDeviceProcessorResult(
                            processedImage = frame,
                            text = null,
                            processingTimeMs = System.currentTimeMillis() - t0
                        )
                    } else {
                        // Online: per-frame homography, degree gating, strip preview
                        return@withContext processOnlineCapture(frame, t0)
                    }
                }

                // ── Fallback: idle without stitch, show raw frame ────────────
                OnDeviceProcessorResult(
                    processedImage = frame,
                    text = null,
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

    /**
     * Reset to IDLE state, clearing any stitched result, without starting a new capture.
     * Used when "Start" is pressed after a completed panorama — the strip/degree view is
     * restored and the user can begin a fresh sweep with the camera button.
     */
    fun resetToIdle() {
        startRequested = false
        stopRequested  = false
        state.reset()                   // clears stitchedResult, hierarchyNodes, etc.
        _hierarchyReady.value    = false
        _hierarchyNodesFlow.value = emptyList()
        Log.d(TAG, "resetToIdle: processor cleared to IDLE")
    }

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
     * Allowed as soon as stitching is complete, even while Florence scene
     * analysis is still running in the background. Hierarchy nodes will
     * populate progressively; the live cutout and panorama are available
     * immediately. No-op if there is no stitched result yet.
     */
    fun enterRealityProxy() {
        if (state.stitchedResult == null) {
            Log.d(TAG, "enterRealityProxy() ignored — no stitched result yet")
            return
        }
        realityProxyRequested = true
        val nodeCount = state.hierarchyNodes.size
        Log.d(TAG, "enterRealityProxy() requested — $nodeCount nodes available so far")
    }

    /**
     * Exit Reality Proxy mode and return to frozen panorama display.
     *
     * Phase is reset immediately rather than via a flag, because
     * finishPanoramaCapture() cancels localProcessingJob before the next
     * process() call can run — so the flag would never be consumed and
     * state.phase would remain REALITY_PROXY, breaking re-entry.
     */
    fun exitRealityProxy() {
        exitProxyRequested = false   // clear any stale flag
        state.phase = PanoramaPhase.IDLE
        localizer?.reset()
        lastAnnouncedNodeLabel = null
        Log.d(TAG, "exitRealityProxy() — phase reset to IDLE immediately")
    }

    /**
     * Load a previously saved panorama bitmap and re-run the Florence scene analysis
     * pipeline on it, exactly as happens after a live sweep.
     *
     * - Sets [state.stitchedResult] to the provided bitmap.
     * - Resets hierarchy and fires [_hierarchyReady] false → true when done.
     * - [StreamViewModel.startHierarchyWatcher] can be called immediately after to
     *   drive UI state in the same way as after a live stitch.
     *
     * Angle values are inferred from pixel width (minAngle=0, maxAngle=width/PX_PER_DEG).
     * This is only used for Reality Proxy localization, which is unavailable without
     * live keyframes; overlay display is purely fraction-based and is unaffected.
     */
    /**
     * Load a previously saved panorama bitmap and re-run the Florence scene analysis
     * pipeline on it, exactly as happens after a live sweep.
     *
     * @param bitmap          Full-resolution stitched panorama.
     * @param glassioKeyframes Original captured keyframes loaded from a `.glassio` sidecar,
     *                        or empty if no sidecar exists. When non-empty, Reality Proxy will
     *                        use keyframe-based localization (much more reliable than strip mode).
     *                        Bitmaps are adopted into [state.keyframes] and recycled on
     *                        the next [PanoramaState.reset].
     */
    fun loadAndAnalyzePanorama(
        bitmap: Bitmap,
        glassioKeyframes: List<Keyframe> = emptyList(),
        precomputedNodes: List<HierarchyNode> = emptyList(),
    ) {
        val copy = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        state.stitchedResult  = copy
        state.hierarchyNodes  = emptyList()
        state.phase           = PanoramaPhase.IDLE
        hierarchyBuilder.setNodes(emptyList())
        _hierarchyReady.value = false

        // Replace any existing keyframes with the glassio-loaded ones.
        state.keyframes.forEach {
            if (!it.bitmap.isRecycled) it.bitmap.recycle()
            if (it.thumbnail !== it.bitmap && !it.thumbnail.isRecycled) it.thumbnail.recycle()
        }
        state.keyframes.clear()
        state.keyframes.addAll(glassioKeyframes)

        // Angular span: derive from keyframe FOV if available, otherwise pixel-width heuristic.
        state.panoramaAngularSpanDeg = if (glassioKeyframes.isNotEmpty()) {
            val kfBmp = glassioKeyframes.first().bitmap
            (copy.width.toFloat() * cameraHFovDeg(kfBmp.width, kfBmp.height) / kfBmp.width)
                .coerceIn(10f, 350f)
        } else {
            (copy.width.toFloat() * 65f / 720f).coerceIn(10f, 350f)
        }
        Log.d(TAG, "loadAndAnalyzePanorama: ${glassioKeyframes.size} glassio keyframes, " +
            "span=${"%.1f".format(state.panoramaAngularSpanDeg)}°")

        // If nodes were already computed (e.g. loaded from .glassio), skip Florence entirely.
        if (precomputedNodes.isNotEmpty()) {
            Log.d(TAG, "loadAndAnalyzePanorama: using ${precomputedNodes.size} precomputed nodes (skipping Florence)")
            hierarchyBuilder.setNodes(precomputedNodes)
            state.hierarchyNodes   = precomputedNodes
            _hierarchyNodesFlow.value = precomputedNodes
            _hierarchyReady.value  = true
            return
        }

        val fp = florenceProcessor
        processorScope.launch {
            var regions = emptyList<Pair<RectF, String>>()
            val nodes: List<HierarchyNode>
            if (fp != null) {
                Log.d(TAG, "loadAndAnalyzePanorama: launching Florence on ${copy.width}×${copy.height}")
                regions = fp.analyzeRegionsTiled(copy)
                nodes = if (regions.isNotEmpty()) {
                    val inferredMaxAngle = copy.width / PX_PER_DEG
                    val rawNodes = hierarchyBuilder.buildFromFlorence(
                        regions, copy.width, copy.height, 0f, inferredMaxAngle, 65f
                    ).also { Log.d(TAG, "loadAndAnalyzePanorama: ${it.size} nodes from Florence") }
                    enrichWithSegmentation(copy, regions, rawNodes)
                } else {
                    Log.d(TAG, "loadAndAnalyzePanorama: Florence returned no regions")
                    emptyList()
                }
            } else {
                Log.d(TAG, "loadAndAnalyzePanorama: no Florence processor — no hierarchy")
                nodes = emptyList()
            }
            hierarchyBuilder.setNodes(nodes)
            state.hierarchyNodes   = nodes
            _hierarchyNodesFlow.value = nodes
            _hierarchyReady.value  = true

            launchDescriptionEnrichment(copy, nodes, regions)
        }
    }


    // ── Description enrichment ───────────────────────────────────────────────

    /**
     * For each node, crop its bbox from [panorama] and run
     * `<MORE_DETAILED_CAPTION>` via Florence. Updates [state.hierarchyNodes]
     * and [_hierarchyNodesFlow] after each node so the ViewModel can re-save
     * the .glassio file incrementally.
     *
     * Pauses automatically while [PanoramaPhase.REALITY_PROXY] is active to
     * avoid starving the live localization frame loop.
     *
     * [regions] must be in the same order as [nodes] (both sorted by centerX
     * as produced by [PanoramaHierarchyBuilder.buildFromFlorence]).
     */
    private fun launchDescriptionEnrichment(
        panorama: Bitmap,
        nodes: List<HierarchyNode>,
        regions: List<Pair<RectF, String>>,
    ) {
        val fp = florenceProcessor ?: return
        if (nodes.isEmpty() || regions.isEmpty()) return
        val sortedBboxes = regions.sortedBy { (bbox, _) -> bbox.centerX() }.map { it.first }

        processorScope.launch {
            val enriched = nodes.toMutableList()
            for (i in nodes.indices) {
                // Suspend while Reality Proxy is live — Florence inference would
                // hold isProcessing for ~6s and drop all localization frames.
                while (state.phase == PanoramaPhase.REALITY_PROXY) { delay(500) }

                val bbox  = sortedBboxes.getOrNull(i) ?: continue
                val cropL = bbox.left.toInt().coerceIn(0, panorama.width - 1)
                val cropT = bbox.top.toInt().coerceIn(0, panorama.height - 1)
                val cropW = bbox.width().toInt().coerceIn(1, panorama.width - cropL)
                val cropH = bbox.height().toInt().coerceIn(1, panorama.height - cropT)
                val crop  = Bitmap.createBitmap(panorama, cropL, cropT, cropW, cropH)
                val desc  = fp.captionImage(crop, detailed = true)
                crop.recycle()

                if (desc != null) {
                    enriched[i] = enriched[i].copy(longDescription = desc)
                    val snapshot = enriched.toList()
                    state.hierarchyNodes  = snapshot
                    _hierarchyNodesFlow.value = snapshot
                    Log.d(TAG, "enriched node $i '${nodes[i].label}': ${desc.take(80)}")
                }
            }
            Log.d(TAG, "description enrichment complete: " +
                "${enriched.count { it.longDescription.isNotEmpty() }}/${nodes.size} nodes enriched")
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /**
     * Run MobileSAM segmentation on each detected region and return a new node list
     * with [HierarchyNode.color] and [HierarchyNode.polygonXY] filled in.
     *
     * SAM encodes [panorama] once, then decodes a mask per bbox (~6ms each).
     * If SAM is unavailable the original nodes are returned with only colors set.
     */
    private suspend fun enrichWithSegmentation(
        panorama: Bitmap,
        regions: List<Pair<RectF, String>>,
        nodes: List<HierarchyNode>,
    ): List<HierarchyNode> {
        val sp = samProcessor
        if (sp == null || !sp.isReady) {
            Log.w(TAG, "enrichWithSegmentation: SAM not ready — using bbox fallback for all nodes")
            return nodes.mapIndexed { i, node -> node.copy(color = NODE_COLORS[i % NODE_COLORS.size]) }
        }
        if (regions.size != nodes.size) {
            Log.w(TAG, "enrichWithSegmentation: size mismatch regions=${regions.size} nodes=${nodes.size} — skipping")
            return nodes
        }

        // Both lists are sorted by x-position (buildFromFlorence sorts by centerX).
        val sortedRegions = regions.sortedBy { (bbox, _) -> bbox.centerX() }
        val bboxes = sortedRegions.map { (bbox, _) -> bbox }

        Log.d(TAG, "enrichWithSegmentation: encoding ${panorama.width}×${panorama.height} panorama…")
        val polygons = sp.segmentRegions(panorama, bboxes)

        return nodes.mapIndexed { i, node ->
            val color = NODE_COLORS[i % NODE_COLORS.size]
            val pts   = polygons.getOrNull(i)
            if (pts == null || pts.size < 6) {
                Log.d(TAG, "enrichWithSegmentation: node $i (${node.label}) — no polygon, bbox fallback")
                node.copy(color = color)
            } else {
                val normalized = FloatArray(pts.size) { j ->
                    if (j % 2 == 0) (pts[j] / panorama.width).coerceIn(0f, 1f)
                    else            (pts[j] / panorama.height).coerceIn(0f, 1f)
                }
                Log.d(TAG, "enrichWithSegmentation: node $i (${node.label}) — ${pts.size / 2} SAM pts")
                node.copy(color = color, polygonXY = normalized)
            }
        }
    }

    /**
     * Launch Florence scene analysis in processorScope. Shared by online stop
     * handler and offline ②b completion handler.
     */
    private fun launchHierarchyAnalysis(
        panorama: Bitmap,
        panoramaWidth: Int,
        keyframeSnapshot: List<Keyframe>,
    ) {
        val fp = florenceProcessor
        processorScope.launch {
            var regions = emptyList<Pair<RectF, String>>()
            val nodes: List<HierarchyNode>
            if (fp != null) {
                Log.d(TAG, "Launching Florence scene analysis on ${panorama.width}×${panorama.height} panorama…")
                regions = fp.analyzeRegionsTiled(panorama)

                nodes = if (regions.isNotEmpty()) {
                    val minAngle = keyframeSnapshot.minOf { it.angleDeg }
                    val maxAngle = keyframeSnapshot.maxOf { it.angleDeg }
                    val fov = keyframeSnapshot.firstOrNull()?.bitmap?.let {
                        cameraHFovDeg(it.width, it.height)
                    } ?: 65f
                    val rawNodes = hierarchyBuilder.buildFromFlorence(
                        regions, panoramaWidth, panorama.height, minAngle, maxAngle, fov
                    ).also { Log.d(TAG, "Florence hierarchy: ${it.size} nodes") }
                    enrichWithSegmentation(panorama, regions, rawNodes)
                } else {
                    Log.d(TAG, "Florence returned no regions — falling back to angle labels")
                    hierarchyBuilder.build(keyframeSnapshot, panoramaWidth)
                }
            } else {
                nodes = hierarchyBuilder.build(keyframeSnapshot, panoramaWidth)
            }
            hierarchyBuilder.setNodes(nodes)
            state.hierarchyNodes  = nodes
            _hierarchyNodesFlow.value = nodes
            _hierarchyReady.value = true
            Log.d(TAG, "Hierarchy ready: ${nodes.size} nodes — signalling ViewModel")

            launchDescriptionEnrichment(panorama, nodes, regions)
        }
    }

    /**
     * Online capture loop: per-frame homography via FeatureTracker, degree-based
     * gating, strip preview overlay. Used when OFFLINE_CAPTURE = false.
     */
    private fun processOnlineCapture(frame: Bitmap, t0: Long): OnDeviceProcessorResult {
        val ft = featureTracker!!
        val H = ft.computeHomography(frame)

        if (H == null) {
            consecutiveStillFrames++
            val now = System.currentTimeMillis()
            val guidance = if (consecutiveStillFrames >= SLOW_THRESHOLD
                && now - lastGuidanceMs > MIN_GUIDANCE_MS
            ) {
                lastGuidanceMs = now
                consecutiveStillFrames = 0
                "Pan slowly to the right"
            } else null
            return OnDeviceProcessorResult(
                processedImage = renderFrame(frame),
                text = guidance,
                processingTimeMs = System.currentTimeMillis() - t0
            )
        }
        consecutiveStillFrames = 0

        val shiftPx  = extractHorizontalShift(H, frame.width, frame.height)
        val shiftDeg = shiftPx / frame.width * cameraHFovDeg(frame.width, frame.height)
        val vertPx   = extractVerticalShift(H, frame.width, frame.height)

        val absDeg = abs(shiftDeg)
        when {
            absDeg > MAX_CAPTURE_DEGREES -> {
                // Too fast — reset reference, discard
                ft.setReferenceFrame(frame, emptyList())
                state.lastAcceptedFrame?.recycle()
                state.lastAcceptedFrame = frame.copy(Bitmap.Config.ARGB_8888, false)
                Log.d(TAG, "Too fast: ${"%.1f".format(shiftDeg)}° — reference reset")
                val now = System.currentTimeMillis()
                val guidance = if (now - lastGuidanceMs > MIN_GUIDANCE_MS) {
                    lastGuidanceMs = now
                    "Too fast, slow down"
                } else null
                return OnDeviceProcessorResult(
                    processedImage = renderFrame(frame),
                    text = guidance,
                    processingTimeMs = System.currentTimeMillis() - t0
                )
            }
            absDeg < MIN_CAPTURE_DEGREES -> {
                // Too close — skip
                state.skippedCount++
                return OnDeviceProcessorResult(
                    processedImage = renderFrame(frame),
                    text = null,
                    processingTimeMs = System.currentTimeMillis() - t0
                )
            }
            else -> {
                // Accepted keyframe
                state.currentAngleDeg += shiftDeg
                state.currentVerticalPx += vertPx
                captureKeyframe(frame, state.currentAngleDeg, state.currentVerticalPx, H)
                state.skippedCount = 0

                ft.setReferenceFrame(frame, emptyList())
                state.lastAcceptedFrame?.recycle()
                state.lastAcceptedFrame = frame.copy(Bitmap.Config.ARGB_8888, false)

                // Milestone TTS
                val absAngle = abs(state.currentAngleDeg)
                val milestoneDeg = (absAngle / MILESTONE_DEG).toInt() * MILESTONE_DEG
                val now = System.currentTimeMillis()
                val announcement = if (milestoneDeg > lastSpokenMilestoneDeg
                    && now - lastGuidanceMs > MIN_GUIDANCE_MS
                ) {
                    lastSpokenMilestoneDeg = milestoneDeg
                    lastGuidanceMs = now
                    "${"%.0f".format(milestoneDeg)} degrees"
                } else null

                return OnDeviceProcessorResult(
                    processedImage = renderFrame(frame),
                    text = announcement,
                    processingTimeMs = System.currentTimeMillis() - t0
                )
            }
        }
    }

    /**
     * Lightweight motion gate: returns true if the frame differs enough from
     * the last captured frame AND enough time has elapsed.
     *
     * Downscales to 64×64 and computes mean absolute per-channel pixel difference.
     * ~1ms total — negligible compared to the old per-frame homography (~50-100ms).
     */
    private fun shouldCapture(frame: Bitmap): Boolean {
        val now = System.currentTimeMillis()
        if (now - lastCaptureTimeMs < MIN_CAPTURE_INTERVAL_MS) return false

        val small = Bitmap.createScaledBitmap(frame, DIFF_THUMB_SIZE, DIFF_THUMB_SIZE, true)
        val prev = state.lastCapturedSmall
        if (prev == null) {
            state.lastCapturedSmall = small
            lastCaptureTimeMs = now
            return true
        }

        val n = DIFF_THUMB_SIZE * DIFF_THUMB_SIZE
        val pixels1 = IntArray(n)
        val pixels2 = IntArray(n)
        prev.getPixels(pixels1, 0, DIFF_THUMB_SIZE, 0, 0, DIFF_THUMB_SIZE, DIFF_THUMB_SIZE)
        small.getPixels(pixels2, 0, DIFF_THUMB_SIZE, 0, 0, DIFF_THUMB_SIZE, DIFF_THUMB_SIZE)

        var totalDiff = 0L
        for (i in 0 until n) {
            val p1 = pixels1[i]; val p2 = pixels2[i]
            totalDiff += abs((p1 shr 16 and 0xFF) - (p2 shr 16 and 0xFF))  // R
            totalDiff += abs((p1 shr 8 and 0xFF) - (p2 shr 8 and 0xFF))    // G
            totalDiff += abs((p1 and 0xFF) - (p2 and 0xFF))                  // B
        }
        val meanDiff = totalDiff.toFloat() / (n * 3)

        if (meanDiff >= PIXEL_DIFF_THRESHOLD) {
            prev.recycle()
            state.lastCapturedSmall = small
            lastCaptureTimeMs = now
            return true
        } else {
            small.recycle()
            return false
        }
    }

    /**
     * Post-hoc processing: compute homographies between consecutive raw frames,
     * derive cumulative angles, and build proper Keyframes for the stitcher.
     *
     * Transfers ownership of accepted raw bitmaps into Keyframes (no extra copy).
     * Rejected raw frames are recycled.
     */
    private fun processFramesPostHoc() {
        val raw = state.rawFrames
        if (raw.size < 2) {
            if (raw.size == 1) {
                addRawAsKeyframe(raw[0], 0f, 0f, IDENTITY_H.copyOf())
            }
            raw.clear()
            return
        }

        val ft = featureTracker ?: run {
            Log.e(TAG, "processFramesPostHoc: no FeatureTracker — aborting")
            raw.forEach { if (!it.isRecycled) it.recycle() }
            raw.clear()
            return
        }

        val rawCount = raw.size
        val accepted = mutableSetOf<Int>()
        var cumulAngle = 0f
        var cumulVertical = 0f
        var consecutiveFailures = 0

        // First frame is always accepted at angle 0
        accepted.add(0)
        addRawAsKeyframe(raw[0], 0f, 0f, IDENTITY_H.copyOf())
        ft.setReferenceFrame(raw[0], emptyList())

        for (i in 1 until rawCount) {
            val H = ft.computeHomography(raw[i])
            if (H == null) {
                consecutiveFailures++
                Log.d(TAG, "Post-hoc: frame $i/$rawCount — null H (consecutive=$consecutiveFailures)")
                // If too many consecutive failures, reference is too stale — reset
                if (consecutiveFailures >= 3) {
                    ft.setReferenceFrame(raw[i], emptyList())
                    consecutiveFailures = 0
                    Log.d(TAG, "Post-hoc: reference reset to frame $i after $consecutiveFailures failures")
                }
                continue
            }
            consecutiveFailures = 0

            val shiftPx = extractHorizontalShift(H, raw[i].width, raw[i].height)
            val shiftDeg = shiftPx / raw[i].width * cameraHFovDeg(raw[i].width, raw[i].height)
            val vertPx = extractVerticalShift(H, raw[i].width, raw[i].height)

            if (abs(shiftDeg) > MAX_CAPTURE_DEGREES) {
                Log.d(TAG, "Post-hoc: frame $i/$rawCount — too fast (${"%.1f".format(shiftDeg)}°), resetting reference")
                ft.setReferenceFrame(raw[i], emptyList())
                continue
            }

            cumulAngle += shiftDeg
            cumulVertical += vertPx
            postHocAngleDeg = cumulAngle  // volatile write — read by process() for progress
            accepted.add(i)
            addRawAsKeyframe(raw[i], cumulAngle, cumulVertical, H)
            ft.setReferenceFrame(raw[i], emptyList())
            Log.d(TAG, "Post-hoc: frame $i/$rawCount accepted — shift=${"%.1f".format(shiftDeg)}° cumul=${"%.1f".format(cumulAngle)}°")
        }

        // Recycle raw frames NOT accepted as keyframes (accepted ones are now owned by Keyframe)
        for (i in 0 until rawCount) {
            if (i !in accepted && !raw[i].isRecycled) raw[i].recycle()
        }
        raw.clear()
        state.lastCapturedSmall?.let { if (!it.isRecycled) it.recycle() }
        state.lastCapturedSmall = null
        state.keyframes.sortBy { it.angleDeg }

        Log.d(TAG, "Post-hoc complete: ${state.keyframes.size} keyframes from $rawCount raw frames, " +
            "span=${"%.1f".format(cumulAngle)}°")
    }

    /**
     * Add a raw frame bitmap as a Keyframe WITHOUT copying it (transfers ownership).
     * Only creates a thumbnail; the full bitmap is reused directly.
     */
    private fun addRawAsKeyframe(bitmap: Bitmap, angleDeg: Float, verticalOffsetPx: Float, H: FloatArray) {
        val thumbH = (bitmap.height * STRIP_HEIGHT_FRACTION).toInt().coerceAtLeast(1)
        val thumbW = (bitmap.width.toFloat() / bitmap.height * thumbH).toInt().coerceAtLeast(1)
        val thumbnail = Bitmap.createScaledBitmap(bitmap, thumbW, thumbH, true)
        state.keyframes.add(Keyframe(bitmap, thumbnail, angleDeg, verticalOffsetPx, H))
    }

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
        //canvas.drawText(statusText, 20f, 60f, statusPaint)

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
