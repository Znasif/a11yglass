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
                        // Compute angular span from the stitched pixel width and frame FOV.
                        val hFov = state.keyframes.firstOrNull()?.bitmap
                            ?.let { cameraHFovDeg(it.width, it.height) } ?: 65f
                        val frameW = state.keyframes.firstOrNull()?.bitmap?.width ?: 720
                        state.panoramaAngularSpanDeg =
                            ((state.stitchedResult!!.width.toFloat() * hFov / frameW)
                                .coerceIn(10f, 350f))
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
                    if (panoramaWidth != null) {
                        val fp = florenceProcessor
                        processorScope.launch {
                            val nodes: List<HierarchyNode>
                            if (fp != null) {
                                Log.d(TAG, "Launching Florence scene analysis on ${panorama.width}×${panorama.height} panorama…")
                                val regions = fp.analyzeRegionsTiled(panorama)

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
                            state.hierarchyNodes = nodes
                            _hierarchyReady.value = true
                            Log.d(TAG, "Hierarchy ready: ${nodes.size} nodes — signalling ViewModel")
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
    fun loadAndAnalyzePanorama(bitmap: Bitmap, glassioKeyframes: List<Keyframe> = emptyList()) {
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

        val fp = florenceProcessor
        processorScope.launch {
            val nodes: List<HierarchyNode>
            if (fp != null) {
                Log.d(TAG, "loadAndAnalyzePanorama: launching Florence on ${copy.width}×${copy.height}")
                val regions = fp.analyzeRegionsTiled(copy)
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
            state.hierarchyNodes  = nodes
            _hierarchyReady.value = true
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
