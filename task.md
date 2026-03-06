# Task: Panorama → Reality Proxy Mode Extension

---

## Phase A — COMPLETED (original three-phase pipeline)

The following has been implemented and is on the `on_device` branch:

- `PanoramaPhase` enum (IDLE, CAPTURING, STITCHING, HIERARCHY_BUILDING, REALITY_PROXY)
- `HierarchyNode.kt` data class
- `PanoramaHierarchyBuilder.kt` — builds from keyframes or Florence regions; `findNodeAt()`
- `PanoramaLocalizer.kt` — absolute xFraction localization against full stitched panorama
- `RealityProxyRenderer.kt` — canvas-based bitmap renderer (to be replaced in Phase B)
- `StripRenderer.kt` — Phase 1 live strip preview (retained as-is)
- `PanoramaProcessor.kt` — full three-phase process() loop, processorScope, @Volatile flags
- `PanoramaSaveManager.kt` — save/load panoramas to external storage
- `FlorenceProcessor.kt` — semantic region detection (WebGPU WebView, correct fp16/q4 dtypes)
- `StreamViewModel.kt` — hierarchy watcher, TTS guidance, capturePhoto() 4-way branch

**Correction from original plan:**
`PanoramaLocalizer` was originally designed as keyframe-chaining (relative). It was implemented
as absolute xFraction localization against the full stitched panorama (matches against a strip
of the panorama bitmap). This is strictly better — failed frames produce stale-but-correct
position rather than accumulated drift. The carousel in Phase B depends on this being absolute.

---

## Phase B — IN PROGRESS: Carousel WebView Renderer

### Goal

Replace all post-stitch rendering (annotated bitmap stitch view, RealityProxyRenderer bitmap)
with a Three.js WebView carousel. The stitched panorama is rendered as the inside of a
cylinder. The camera sits at the origin. PanoramaLocalizer drives cylinder rotation so the
correct part of the panorama is always centred. Depth-Pro displacement is stubbed until the
pipeline is available.

Node overlays become native Android accessibility buttons positioned by angle — not canvas
drawing. This makes the panorama fully navigable by blind users via TalkBack swipe gestures.

### Rendering Phase Map

| Phase | Renderer | Notes |
|---|---|---|
| CAPTURING | StripRenderer → Bitmap | Unchanged — still overlaid on live frame |
| STITCHING | Loading indicator (UI state only) | No bitmap output |
| IDLE (post-stitch) | WebView carousel, static rotation | Replaces annotated bitmap |
| REALITY_PROXY | WebView carousel, localizer-driven rotation | Same WebView, localization on |
| PANORAMA_DONE | WebView carousel, static rotation | Loaded saved panorama |

IDLE and REALITY_PROXY are the same WebView — the only difference is whether
`CarouselWebViewManager.setAngle()` is being called each frame.

### Layer Stack (all post-stitch phases)

```
Compose Box (full frame)
├── AndroidView → WebView          Three.js carousel, full frame, pinch-to-zoom inside JS
└── Box overlay (native Compose)
    ├── Image                      Live camera crop, centre-anchored, fixed position
    ├── CrosshairView              Centre decoration
    ├── NodeButtonRow              Accessible buttons positioned horizontally by angle
    └── NodeDetailPanel            Bottom 22%, shown when a node is selected
```

### New Files

---

#### `app/src/main/assets/panorama_carousel.html`

Single self-contained HTML. Three.js via importmap CDN (same pattern as
`florence_inference.html`). No file system access needed.

**Three.js setup:**
- `CylinderGeometry(10, 10, 8, 512, 128, true)` — 512×128 segments for smooth depth
  displacement; `openEnded: true`
- `geometry.rotateX(Math.PI / 2)` — cylinder axis aligned to Z
- `MeshStandardMaterial { side: THREE.BackSide }` — renders inside of cylinder
- Camera at origin, `camera.up = (0, 0, 1)`, looking along −Y
- `AmbientLight` only — no shadows needed
- `displacementMap` slot present, `displacementScale = -2.0` (white = near, pushed inward)
- Pinch-to-zoom via `touchstart`/`touchmove` on `renderer.domElement` — no OrbitControls,
  no Kotlin bridge involvement. `camera.fov` clamped 20–110°.

**JavaScript API (called from Kotlin via `evaluateJavascript`):**

```javascript
loadPanorama(base64DataUrl)    // set cylinder colour texture; resets zoom to default FOV
loadDepthMap(base64DataUrl)    // set displacement texture; no-op until Depth-Pro ready
setAngle(radians)              // carousel.rotation.z = radians
setZoom(fovDeg)                // camera.fov = clamp(fovDeg, 20, 110); updateProjectionMatrix
```

---

#### `processor/panorama/CarouselWebViewManager.kt`

Owns WebView lifecycle and the Kotlin↔JS bridge.

```kotlin
class CarouselWebViewManager {

    fun initialize(context: Context, webView: WebView)
    // WebSettings: javaScriptEnabled, hardwareAccelerated
    // addJavascriptInterface(this, "Android")
    // loadUrl("file:///android_asset/panorama_carousel.html")

    fun loadPanorama(bitmap: Bitmap)
    // Encode to base64 PNG on IO dispatcher
    // evaluateJavascript("loadPanorama('data:image/png;base64,...')", null)

    fun loadDepthMap(bitmap: Bitmap)
    // Same as loadPanorama but calls loadDepthMap JS function

    fun setAngle(angleDeg: Float)
    // evaluateJavascript("setAngle(${Math.toRadians(angleDeg.toDouble())})", null)

    @JavascriptInterface
    fun onNodeTapped(nodeIndex: Int)
    // Called from JS if node sprites are tapped directly in WebGL (optional path)
}
```

---

#### `ui/panorama/CarouselOverlayState.kt`

Pure computation — no Android dependencies.

```kotlin
data class VisibleNode(
    val node: HierarchyNode,
    val screenXFraction: Float,   // 0..1 across screen width
    val stackDepth: Int,          // vertical stacking offset for overlapping nodes
)

fun computeVisibleNodes(
    nodes: List<HierarchyNode>,
    currentAngleDeg: Float,
    hFovDeg: Float,
): List<VisibleNode>
```

**Visibility:** a node is in view when its angular range overlaps the current FOV:
```
nodeMinAngle < currentAngle + hFov/2  &&  nodeMaxAngle > currentAngle - hFov/2
```
where `nodeMinAngle = node.angleDeg - node.panoramaWidthFraction * totalSpan / 2`

**Screen X fraction:**
```
screenXFraction = (node.angleDeg - currentAngle) / hFov + 0.5f  // clamped 0..1
```

**Overlap:** nodes within 0.08 screen-fraction of each other get increasing `stackDepth`
values → small vertical offset in the overlay (48dp per depth level) so they do not occlude.

**Ordering:** list is sorted by `node.angleDeg` so TalkBack linear swipe traverses
spatially left-to-right without any custom accessibility configuration.

---

### Modified Files

---

#### `processor/panorama/PanoramaProcessor.kt`

- **Remove** `renderRealityProxy()` — no longer produces a bitmap for post-stitch phases
- **Remove** `buildAnnotatedPanorama()` — node overlays are now native buttons
- **Remove** `realityProxyRenderer` field and import
- Post-stitch phases (IDLE, REALITY_PROXY, PANORAMA_DONE) return the raw frame or null
  from `process()` — the image preview is no longer updated during these phases
- **Add** `localizedAngleDeg` as `MutableStateFlow<Float>` (updated each frame in
  REALITY_PROXY loop) so StreamViewModel can collect it and drive `setAngle()`
- `stitchedResult` remains accessible via property for WebView loading
- Everything else (localizer, hierarchyBuilder, processorScope, flags) unchanged

```kotlin
// New StateFlow exposure
private val _localizedAngleDeg = MutableStateFlow(0f)
val localizedAngleDeg: StateFlow<Float> = _localizedAngleDeg.asStateFlow()

// In REALITY_PROXY loop, replace renderRealityProxy() call with:
val xFraction = localizer?.localize(frame, featureTracker!!)
if (xFraction != null) {
    state.localizedAngleDeg = minAngle + xFraction * (maxAngle - minAngle)
    _localizedAngleDeg.value = state.localizedAngleDeg
}
state.focusedNode = hierarchyBuilder.findNodeAt(state.localizedAngleDeg)
return@withContext OnDeviceProcessorResult(
    processedImage = frame,   // raw frame — UI ignores it during REALITY_PROXY
    text = if (state.focusedNode?.label != lastAnnouncedLabel) state.focusedNode?.label else null,
    processingTimeMs = elapsed
)
```

---

#### `StreamViewModel.kt`

- Collect `panoramaProcessor.localizedAngleDeg` → `carouselManager.setAngle()`
- After stitch completes (hierarchy watcher): `carouselManager.loadPanorama(stitchedResult)`
- Compute `visibleNodes` from `hierarchyNodes + localizedAngleDeg` each frame →
  expose as `StateFlow<List<VisibleNode>>`
- `hFovDeg` computed once from first keyframe bitmap dimensions via `cameraHFovDeg()`
- `startHierarchyWatcher()` TTS announcements unchanged
- `loadAndAnalyzePanorama()` path: same — after loading, call `carouselManager.loadPanorama()`

---

#### `ui/StreamScreen.kt`

```kotlin
when (phase) {
    PanoramaPhase.CAPTURING -> {
        // Existing bitmap Image preview — unchanged
        Image(bitmap = processedFrame, ...)
    }
    else -> {
        // Post-stitch: WebView + overlay
        Box(Modifier.fillMaxSize()) {
            AndroidView(factory = { carouselWebView }, Modifier.fillMaxSize())
            LiveCropImage(frame = latestFrame, Modifier.align(Alignment.Center))
            CrosshairView(Modifier.align(Alignment.Center))
            NodeButtonRow(
                visibleNodes = visibleNodes,
                onNodeSelected = { viewModel.selectNode(it) }
            )
            NodeDetailPanel(
                node = selectedNode,
                Modifier.align(Alignment.BottomCenter)
            )
        }
    }
}
```

---

### Node Buttons — Accessibility

```kotlin
@Composable
fun NodeButtonRow(visibleNodes: List<VisibleNode>, onNodeSelected: (HierarchyNode) -> Unit) {
    // visibleNodes already sorted by angle = left-to-right TalkBack order
    visibleNodes.forEach { vn ->
        Box(
            Modifier
                .fillMaxWidth()
                .offset(
                    x = (vn.screenXFraction * screenWidth - buttonHalfWidth).dp,
                    y = (vn.stackDepth * 48).dp
                )
                .semantics {
                    role = Role.Button
                    contentDescription = buildString {
                        append(vn.node.label)
                        if (vn.node.description.isNotEmpty()) {
                            append(". ")
                            append(vn.node.description)
                        }
                    }
                }
                .clickable { onNodeSelected(vn.node) }
        ) {
            // Visual: small pill label, semi-transparent colored background
        }
    }
}
```

TalkBack swipe-right/left traverses nodes in angle order automatically.
Double-tap triggers `selectNode()` → expands detail panel + Gemini Q&A if description empty.
When carousel rotates and a node exits the FOV it is removed from composition →
TalkBack focus moves to the next in-FOV node naturally.

---

### Zoom

Handled entirely inside `panorama_carousel.html` via touch events — no Kotlin bridge.
Pinch-in → decrease FOV (zoom in), pinch-out → increase FOV (zoom out).
`loadPanorama()` resets FOV to default (75°).

---

### Depth Map (deferred)

`loadDepthMap(base64)` is present in the HTML but wired to a no-op until Depth-Pro pipeline:

1. Run Depth-Pro on `stitchedResult` after hierarchy building completes (processorScope)
2. Normalise output to 0–1 greyscale
3. `carouselManager.loadDepthMap(depthBitmap)`

No other changes required — displacement slot already in the Three.js material.

---

### What Is Removed

| Item | Reason |
|---|---|
| `RealityProxyRenderer.kt` | Replaced by Three.js carousel |
| `PanoramaProcessor.buildAnnotatedPanorama()` | Nodes are native accessibility buttons |
| `PanoramaProcessor.renderRealityProxy()` | No bitmap rendering post-stitch |
| Bitmap output for IDLE / REALITY_PROXY / PANORAMA_DONE | WebView owns display |

### What Is Unchanged

| Component | Notes |
|---|---|
| `StripRenderer.kt` | Phase 1 capture preview — bitmap overlay on live frame |
| `PanoramaLocalizer.kt` | Absolute xFraction localization — drives carousel rotation |
| `PanoramaStitcher.kt` | Unchanged |
| `PanoramaHierarchyBuilder.kt` | Unchanged |
| `PanoramaSaveManager.kt` | Loaded panoramas enter PANORAMA_DONE → WebView |
| `FlorenceProcessor.kt` | Unchanged |
| Phase 1 guidance TTS | Unchanged |

---

### Implementation Order

1. `panorama_carousel.html` — Three.js carousel with full JS API + pinch zoom
2. `CarouselWebViewManager.kt` — WebView setup, bitmap loading, angle bridge
3. `PanoramaProcessor.kt` — expose `localizedAngleDeg` StateFlow, remove bitmap rendering paths
4. `StreamViewModel.kt` — collect angle, compute visibleNodes, load panorama into manager
5. `StreamScreen.kt` — swap post-stitch display to WebView + overlay composables
6. `CarouselOverlayState.kt` + node button composables with accessibility semantics
7. Delete `RealityProxyRenderer.kt`

Steps 1 and 2 are independent. Step 3 can proceed in parallel with 1–2. Steps 4–6 depend
on 1–3. Step 7 is last.
