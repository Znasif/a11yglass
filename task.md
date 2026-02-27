# Task: Panorama → Reality Proxy Mode Extension

## Goal

Extend `PanoramaProcessor` from a two-phase pipeline (capture + stitch) into a **three-phase pipeline**:

```
Phase 1: CAPTURE        → user sweeps camera, keyframes collected via SuperPoint+LightGlue
Phase 2: STITCH         → post-hoc composite panorama (existing, unchanged)
Phase 3: REALITY PROXY  → interactive: live center-square tracks position in panorama,
                          pre-derived hierarchy revealed on gaze
```

After stitching completes, pressing the camera button enters Reality Proxy mode. The stitched panorama is shown full-screen with the object hierarchy overlaid as labeled bounding regions. A center reticle tracks where the live camera is pointing by matching each live frame against the stored keyframes (HLOC-style, using the existing SuperPoint+LightGlue `FeatureTracker`). The region under the reticle is highlighted and its description panel is shown at the bottom. No hand gestures — center square is the sole interaction modality.

---

## State Machine

Add to `PanoramaState.kt`:

```kotlin
enum class PanoramaPhase { IDLE, CAPTURING, STITCHING, HIERARCHY_BUILDING, REALITY_PROXY }
```

Full transition graph:

```
startPanorama()
  IDLE ─────────────────────► CAPTURING
                                  │
                            stopPanorama()
                                  │
                              STITCHING  (synchronous in process(), existing)
                                  │
                    auto: launch hierarchyBuilder.build() in processorScope
                                  │
                          HIERARCHY_BUILDING  (async, processorScope)
                                  │
                    auto: nodes ready → state.hierarchyNodes populated
                                  │
                              IDLE + stitchedResult ≠ null
                                  │
                          capturePhoto() (2nd press)
                                  │
                          REALITY_PROXY  (live frame loop)
                                  │
                          capturePhoto() (3rd press)
                                  │
                              IDLE (reset, or keep stitched frozen)
```

---

## New Files  (all in `processor/panorama/`)

---

### 1. `HierarchyNode.kt`

Data class for one region in the pre-derived panorama hierarchy.

```kotlin
data class HierarchyNode(
    val label: String,
    val angleDeg: Float,              // angular center in panorama space
    val panoramaXFraction: Float,     // normalized 0–1 horizontal center in stitched bitmap
    val panoramaWidthFraction: Float, // normalized 0–1 width in stitched bitmap
    val keyframeIndex: Int,           // index into state.keyframes
    val description: String = "",     // filled async (FastVLM) or left as angle label for v1
    val children: List<HierarchyNode> = emptyList()
)
```

Level-0 node = root (full panorama).
Level-1 nodes = one per keyframe (the angular slice that keyframe covers).
Level-2 (future) = sub-object detection within each keyframe crop.

---

### 2. `PanoramaHierarchyBuilder.kt`

Builds and queries the hierarchy tree from keyframes right after stitching.

```kotlin
class PanoramaHierarchyBuilder {

    // Populated once, then read-only during REALITY_PROXY
    @Volatile private var nodes: List<HierarchyNode> = emptyList()

    /**
     * Build hierarchy from keyframes.
     * Called from processorScope (background thread) immediately after stitch.
     * panoramaWidth: width of the stitched bitmap in px.
     */
    fun build(keyframes: List<Keyframe>, panoramaWidth: Int): List<HierarchyNode>

    /** Swap in the completed list (atomic write). */
    fun setNodes(n: List<HierarchyNode>) { nodes = n }

    /** Query: which node's angular range contains angleDeg? */
    fun findNodeAt(angleDeg: Float): HierarchyNode?

    fun clear() { nodes = emptyList() }
}
```

**`build()` algorithm:**

1. `totalSpan = maxAngle - minAngle` (from keyframes)
2. For each keyframe `kf`:
   - `xFraction = (kf.angleDeg - minAngle) / totalSpan`
   - `widthFraction = CAMERA_FOV_DEGREES / totalSpan`
   - `label = "${kf.angleDeg.toInt()}°"`  (v1 — no FastVLM yet)
   - Construct `HierarchyNode(label, kf.angleDeg, xFraction, widthFraction, index)`
3. Return list sorted by `angleDeg`

**`findNodeAt()` algorithm:**

Linear scan; return the node `n` where `|n.angleDeg - angleDeg| < CAMERA_FOV_DEGREES / 2f`.
Ties broken by closest center.

**v2 (deferred):** After `build()` returns, launch a separate coroutine that iterates nodes,
writes each `keyframe.bitmap` to a temp file, calls FastVLM Engine (the same approach as
`SceneDescriptionProcessor`), and updates the node's `description` field. The renderer
shows whatever is in `description` at render time, so descriptions populate progressively.

---

### 3. `PanoramaLocalizer.kt`

HLOC-style live-to-panorama localization using the shared `FeatureTracker`.

```kotlin
class PanoramaLocalizer {

    private var currentKeyframeIdx = 0
    private var referenceAngleDeg  = 0f

    /**
     * Set initial reference keyframe (call once on entering REALITY_PROXY).
     * Uses the midpoint keyframe as starting estimate.
     */
    fun initialize(keyframes: List<Keyframe>, featureTracker: FeatureTracker)

    /**
     * Match liveFrame against the current reference keyframe.
     * Returns estimated panorama angle (degrees) of the live frame center, or null on failure.
     * Side-effect: advances currentKeyframeIdx and updates featureTracker reference if the
     * estimated center has drifted outside the current keyframe's angular range.
     */
    fun localize(
        liveFrame: Bitmap,
        keyframes: List<Keyframe>,
        featureTracker: FeatureTracker
    ): Float?

    fun reset() { currentKeyframeIdx = 0; referenceAngleDeg = 0f }
}
```

**`localize()` algorithm:**

1. `H = featureTracker.computeHomography(liveFrame)` — null → return null
2. `shiftPx = extractHorizontalShift(H, liveFrame.width, liveFrame.height)`
   *(reuse the same math as `PanoramaProcessor.extractHorizontalShift`; move to a
   package-level function in `PanoramaState.kt` so both classes can call it)*
3. `estimatedAngle = referenceAngleDeg - shiftPx / liveFrame.width * CAMERA_FOV_DEGREES`
   *(negative because camera panning right → scene shifts left → angle increases)*
4. Boundary check: if `estimatedAngle` is more than `CAMERA_FOV_DEGREES * 0.6f` away from
   `referenceAngleDeg`, find the nearest keyframe by angle, call
   `featureTracker.setReferenceFrame(keyframes[newIdx].bitmap, emptyList())`, update
   `currentKeyframeIdx` and `referenceAngleDeg`.
5. Return `estimatedAngle` (clamped to `[minAngle, maxAngle]`)

**Why this works:** `FeatureTracker` was designed for inter-frame matching in the panorama
sweep; it accepts any two bitmaps. After Phase 1 ends, the tracker's reference is simply
switched to a stored keyframe instead of the last captured live frame. The math for
extracting angular shift from H is identical to Phase 1.

---

### 4. `RealityProxyRenderer.kt`

Stateless Canvas renderer for the Reality Proxy overlay.

```kotlin
class RealityProxyRenderer {

    /**
     * Composite the panorama + hierarchy overlay onto a copy of [frame] and return it.
     *
     * @param frame           Current live camera frame (used for output bitmap dimensions only)
     * @param panorama        Stitched panorama bitmap
     * @param nodes           Pre-built hierarchy nodes
     * @param currentAngleDeg Current estimated angle from PanoramaLocalizer
     * @param minAngleDeg     Min angle in the panorama
     * @param maxAngleDeg     Max angle in the panorama
     * @param focusedNode     Hierarchy node under the center square, or null
     */
    fun render(
        frame: Bitmap,
        panorama: Bitmap,
        nodes: List<HierarchyNode>,
        currentAngleDeg: Float,
        minAngleDeg: Float,
        maxAngleDeg: Float,
        focusedNode: HierarchyNode?
    ): Bitmap
}
```

**Render pipeline:**

1. **Create output** = `frame.copy(ARGB_8888, true)` (same resolution as live feed)
2. **Fit panorama** into output with letterbox:
   - `scaleX = outW / panorama.width`, `scaleY = outH / panorama.height`
   - `scale = min(scaleX, scaleY)`
   - `drawW = panorama.width * scale`, `drawH = panorama.height * scale`
   - `offsetX = (outW - drawW) / 2f`, `offsetY = (outH - drawH) / 2f`
   - Draw via `canvas.drawBitmap(panorama, null, RectF(offsetX, offsetY, offsetX+drawW, offsetY+drawH), null)`
3. **Draw hierarchy nodes** (gray outline rectangles + labels):
   - For each node: `screenLeft = offsetX + node.panoramaXFraction * drawW - node.panoramaWidthFraction * drawW / 2`
   - `screenRight = screenLeft + node.panoramaWidthFraction * drawW`
   - Draw `RectF(screenLeft, offsetY, screenRight, offsetY + drawH)` with gray stroke paint
   - Draw label at `(screenLeft + 8, offsetY + 40)`
4. **Draw FOV indicator** (blue fill + white border):
   - `fovSpan = CAMERA_FOV_DEGREES / (maxAngleDeg - minAngleDeg) * drawW`
   - `fovCenterX = offsetX + (currentAngleDeg - minAngleDeg) / (maxAngleDeg - minAngleDeg) * drawW`
   - `RectF(fovCenterX - fovSpan/2, offsetY, fovCenterX + fovSpan/2, offsetY + drawH)`
5. **Draw center crosshair** at `(fovCenterX, offsetY + drawH/2)` — 40px arms, white, 2.5px stroke
6. **Highlight focused node** (if non-null):
   - Redraw its rect with bright cyan stroke + semi-transparent cyan fill
7. **Detail panel** (bottom strip, ~22% of outH):
   - Dark semi-transparent background across full width
   - `focusedNode?.label` in large white bold text
   - `focusedNode?.description` (or `"Look at a region to see details"` if null) in smaller text
   - If `description` is empty (v1): show `"${focusedNode.angleDeg.toInt()}° — tap to describe (coming soon)"`

---

## Modified Files

---

### `PanoramaState.kt`

1. Add `PanoramaPhase` enum (see above).
2. Add fields to `PanoramaState`:
   ```kotlin
   var phase: PanoramaPhase = PanoramaPhase.IDLE
   @Volatile var hierarchyNodes: List<HierarchyNode> = emptyList()
   var localizedAngleDeg: Float = 0f
   var focusedNode: HierarchyNode? = null
   ```
3. Update `reset()`:
   ```kotlin
   phase = PanoramaPhase.IDLE
   hierarchyNodes = emptyList()
   localizedAngleDeg = 0f
   focusedNode = null
   ```
4. Move `extractHorizontalShift()` from `PanoramaProcessor` to a package-level function
   here (or a companion object), making it accessible to `PanoramaLocalizer` without
   duplicating the math.

---

### `PanoramaProcessor.kt`

**New sub-components (initialized in `initialize()`):**
```kotlin
private val hierarchyBuilder = PanoramaHierarchyBuilder()
private var localizer: PanoramaLocalizer? = null
private val realityProxyRenderer = RealityProxyRenderer()
private val processorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
```

**New @Volatile control flags:**
```kotlin
@Volatile private var realityProxyRequested    = false
@Volatile private var exitRealityProxyRequested = false
```

**`initialize()` extension:**
```kotlin
localizer = PanoramaLocalizer()
```

**`process()` — new blocks inserted in order:**

```
① After startRequested handler (unchanged)

② After stopRequested handler — extend:
   // Existing: stitcher.stitch() → state.stitchedResult
   // NEW after stitch:
   state.phase = PanoramaPhase.IDLE
   processorScope.launch {
       val nodes = hierarchyBuilder.build(state.keyframes, state.stitchedResult!!.width)
       hierarchyBuilder.setNodes(nodes)
       state.hierarchyNodes = nodes
       Log.d(TAG, "Hierarchy built: ${nodes.size} nodes")
   }
   // return existing OnDeviceProcessorResult (unchanged)

③ NEW — handle realityProxyRequested (before normal capture loop):
   if (realityProxyRequested) {
       realityProxyRequested = false
       state.phase = PanoramaPhase.REALITY_PROXY
       localizer?.initialize(state.keyframes, featureTracker!!)
       return@withContext OnDeviceProcessorResult(
           processedImage = realityProxyRenderer.render(frame, state.stitchedResult!!, ...),
           text = "Reality Proxy — look at objects",
           processingTimeMs = elapsed
       )
   }

④ NEW — handle exitRealityProxyRequested:
   if (exitRealityProxyRequested) {
       exitRealityProxyRequested = false
       state.phase = PanoramaPhase.IDLE
       localizer?.reset()
       return@withContext OnDeviceProcessorResult(
           processedImage = state.stitchedResult ?: frame,
           text = null,
           processingTimeMs = elapsed
       )
   }

⑤ MODIFY existing "idle with stitchedResult" block:
   if (!state.isCapturing && state.stitchedResult != null
       && state.phase != PanoramaPhase.REALITY_PROXY) {
       // existing frozen-result return, add status text:
       val statusText = if (state.hierarchyNodes.isEmpty()) "Building scene map…" else null
       return@withContext OnDeviceProcessorResult(
           processedImage = state.stitchedResult,
           text = statusText,
           processingTimeMs = elapsed
       )
   }

⑥ NEW — Reality Proxy live frame loop:
   if (state.phase == PanoramaPhase.REALITY_PROXY && state.stitchedResult != null) {
       val angle = localizer?.localize(frame, state.keyframes, featureTracker!!)
                   ?: state.localizedAngleDeg
       state.localizedAngleDeg = angle
       state.focusedNode = hierarchyBuilder.findNodeAt(angle)
       return@withContext OnDeviceProcessorResult(
           processedImage = realityProxyRenderer.render(
               frame, state.stitchedResult!!,
               state.hierarchyNodes, angle,
               state.keyframes.minOf { it.angleDeg },
               state.keyframes.maxOf { it.angleDeg },
               state.focusedNode
           ),
           text = state.focusedNode?.label,
           processingTimeMs = elapsed
       )
   }
```

**New public API:**
```kotlin
fun enterRealityProxy() {
    if (state.stitchedResult != null) realityProxyRequested = true
}
fun exitRealityProxy() { exitRealityProxyRequested = true }
val phase: PanoramaPhase get() = state.phase
val hasStitchedResult: Boolean get() = state.stitchedResult != null
```

**`release()` extension:**
```kotlin
processorScope.cancel()
localizer?.reset()
hierarchyBuilder.clear()
```

---

### `StreamViewModel.kt` — `capturePhoto()` panorama branch

Replace the current `if (processor.isCapturing) … else …` with a 4-way `when`:

```kotlin
if (processor is PanoramaProcessor) {
    when {
        processor.isCapturing -> {
            // ── Stop sweep (existing) ────────────────────────────────────────
            processor.stopPanorama()
            _uiState.update { it.copy(statusMessage = "Stitching panorama…") }
        }

        processor.phase == PanoramaPhase.REALITY_PROXY -> {
            // ── Exit Reality Proxy ───────────────────────────────────────────
            processor.exitRealityProxy()
            // Keep stitchedResult frozen; stop processing job
            finishPanoramaCapture()
            _uiState.update { it.copy(statusMessage = "Panorama ready — tap camera to enter proxy mode") }
        }

        processor.hasStitchedResult -> {
            // ── Enter Reality Proxy ──────────────────────────────────────────
            if (!_uiState.value.isStreamingToServer) startServerStreaming()
            processor.enterRealityProxy()
            _uiState.update { it.copy(statusMessage = "Reality Proxy — look at objects") }
        }

        else -> {
            // ── Start new sweep (existing) ────────────────────────────────────
            _uiState.update { it.copy(processedFrame = null) }
            System.gc()
            if (!_uiState.value.isStreamingToServer) startServerStreaming()
            processor.startPanorama()
            _uiState.update { it.copy(statusMessage = "Panorama sweep started — pan slowly") }
        }
    }
    return
}
```

The existing `handleLocalProcessorResult()` check for `"Panorama complete"` that calls
`finishPanoramaCapture()` remains unchanged — it correctly stops processing after stitching
so the frozen panorama is displayed while hierarchy building runs in `processorScope`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| FeatureTracker reuse across phases | Avoids second ONNX session; same 4-thread XNNPACK config; only the reference bitmap changes |
| Hierarchy building in `processorScope` (not `localProcessingJob`) | `localProcessingJob` is cancelled after stitching; `processorScope` runs independently, outlives the job |
| Phase flag `@Volatile`, request flags `@Volatile` | Same pattern as `startRequested`/`stopRequested`; single writer (process coroutine) reads them at the top of each call, no lock needed |
| Horizontal localization only (v1) | Panorama is a horizontal sweep; vertical drift is corrected in Phase 2 stitch but not tracked for gaze |
| Null localization → hold last known angle | Prevents UI jitter if tracker momentarily fails; graceful degradation |
| `extractHorizontalShift` moved to package-level | Eliminates duplication between `PanoramaProcessor` and `PanoramaLocalizer`; pure function, no state |
| FastVLM descriptions deferred (v2) | LiteRT-LM Engine is ~1.1 GB and may not be initialized; v1 labels are angle strings which convey spatial meaning |

---

## Implementation Order

1. `PanoramaPhase` enum + new fields in `PanoramaState.kt`; move `extractHorizontalShift` / `extractVerticalShift` to package-level
2. `HierarchyNode.kt`
3. `PanoramaHierarchyBuilder.kt` (build + findNodeAt, no FastVLM yet)
4. `PanoramaLocalizer.kt`
5. `RealityProxyRenderer.kt`
6. Modify `PanoramaProcessor.kt` (add components, extend process(), new public API)
7. Modify `StreamViewModel.kt` capturePhoto() panorama branch

Steps 2–5 are independent and can be written in parallel. Step 6 depends on all four.
Step 7 depends on step 6.
