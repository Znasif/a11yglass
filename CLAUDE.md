# Reality Proxy — Implementation Reference

**Paper:** "Reality Proxy: Fluid Interactions with Real-World Objects in MR via Abstract Representations"
UIST '25, Liu et al. (NYU / University of Minnesota / Google)

---

## Concept

Reality Proxy decouples user interaction from the physical constraints of real-world objects (distance, size, occlusion, crowding) by creating **proxies** — abstract, manipulable digital representations placed near the user's hand. Selecting a proxy is functionally equivalent to selecting the physical object. Proxies are enriched with AI-derived semantic and hierarchical structure, enabling interactions that are impossible with direct pointing (raycasting).

The system extends the **Gaze+Pinch** interaction paradigm without adding new gestures or vocabulary.

---

## Three-Step Pipeline

```
ACTIVATE  →  GENERATE  →  INTERACT
(Detect)     (Place)       (Manipulate)
```

### Step 1 — Activate: Hierarchical Scene Parsing

Triggered when the user executes a pinch gesture while gazing at an area.

1. **Gaze Region Extension**: The user's gaze region is slightly expanded (e.g., 20 cm threshold) to capture contextual objects just outside direct focus.
2. **Level 1 Object Detection**: A video frame of the extended gaze region is sent to an open-vocabulary object detector (paper uses DINO-X; implementation uses Gemini Vision). The model returns:
   - 2D bounding boxes
   - Pixel masks
   - Semantic labels
   Bounding boxes are sorted by descending area; duplicates with IoU > 0.75 are removed. Remaining detections are **Level 1** objects.
3. **Recursive Sub-Component Detection**: Each Level 1 object's image crop is sent back to the detector to find finer sub-components (**Level 2** objects). This recurses until no new objects appear or the bounding box falls below a minimum size threshold.
4. **Attribute Extraction**: Cropped images of each detected object are sent to an LLM (paper uses GPT-4o; implementation uses Gemini) which returns structured JSON attributes:
   ```json
   {
     "name": "Oven",
     "type": "Appliance",
     "material": "Stainless Steel",
     "color": "Silver",
     "components": ["Handle", "Door", "Control Panel"]
   }
   ```
5. **Output**: A hierarchically structured scene graph (JSON tree) encoding containment relationships and semantic attributes for every detected object.

**Implementation classes:** `Gemini2DBoundingBoxDetector`, `GeminiRaycast`, `SceneObjectManager`, `GeminiGeneral` (base), `GeminiAPI`

**Key consideration:** Object detection runs continuously before the pinch; attribute extraction runs asynchronously after.

---

### Step 2 — Generate: Proxy Layout with Spatial Coherence

Translates the 2D scene graph into 3D manipulable proxy objects placed near the user's hand.

#### 3D Position via Raycasting

- A ray is cast from the user's head (or camera) through the center of each 2D bounding box.
- The ray's intersection with the scene mesh (provided by the device's spatial understanding system) approximates the object's 3D world position.
- This avoids needing an explicit 3D object detector while remaining lightweight.

**Implementation class:** `GeminiRaycast` — uses camera intrinsic/extrinsic matrices to project 2D boxes into 3D rays; intersects with scene mesh; registers results with `SceneObjectManager`.

#### Constraint-Based Layout Optimization

Raw 3D positions would produce impractically large or sparse proxy layouts. Instead:

1. Extract **relative spatial constraints** from the 3D positions: e.g., "Object A is to the left of Object B" → `x_A < x_B`.
2. Run a **constraint satisfaction** optimization (paper uses Z3 solver) that minimizes deviations from inter-object distances while enforcing at least 0.5 cm spacing between all proxies.
3. The result is a rescaled layout that preserves relative spatial topology but fits comfortably near the user's hand.

Each proxy is represented as a fixed-size rectangular 3D object (physical size is irrelevant — proxies are for interaction, not measurement).

#### Lazy-Follow Placement

Proxies are anchored near the user's hand using a **lazy-follow** mechanism:
- While the hand remains within a threshold radius, proxies stay stationary.
- If the hand moves beyond that radius, proxies smoothly follow to stay within reach.
- This eliminates the need to look down to find proxies and reduces jitter from minor hand movement.

**Implementation class:** Uses Unity's LazyFollow package (`unity.xr.interaction.toolkit@2.x`).

---

### Step 3 — Interact: Proxy-Based Manipulation

Proxies support the following interaction vocabulary, all using familiar direct-manipulation gestures:

| Interaction | Gesture | Description |
|---|---|---|
| **Skim & Preview** | Tap + slide across proxies | Hover finger across multiple proxies; each reveals an attribute panel. Efficiently browse many objects without committing to a selection. |
| **Single Select** | Pinch proxy | Selects one object; highlights both proxy and physical object; shows detail panel and AI-generated questions. |
| **Multi-Select (Brush)** | Two-hand pinch → spread | Initiate a selection region; expand with hands apart; all proxies inside the region are selected and highlighted. |
| **Filter by Attribute** | Pinch+hold a proxy → slide to attribute | Pins attribute panel; sliding to an attribute value selects all objects sharing that value. |
| **Physical Affordance** | Touch/drag on a physical surface | When proxies are placed on a physical surface (e.g., table), the surface becomes a touchpad — drag, spread, and pinch directly on the surface. |
| **Semantic Grouping** | Double-tap a proxy | Groups all proxies sharing the same attribute value with a highlight color; re-represents them as a group proxy. |
| **Spatial Zoom** | Two-hand zoom gesture | Navigates the spatial hierarchy — zoom in to see sub-components (Level 2 objects) or zoom out to see containing objects (Level 1). |
| **Custom Group** | Brush gesture in empty space | Creates a persistent cubic container; pinch+hold the container while tapping proxies to clone them in. Two-hand zoom shrinks the group to a single proxy supporting all single-object operations. |
| **Object Comparison** | Select exactly 2 objects | Triggers AI comparison generation: structured similarities, differences, and functional relationships displayed in a floating panel. |

**Implementation classes:** `SphereToggleScript`, `MultiSelectManager`, `ObjectComparisonManager`, `ComparisonPanel`, `RelationshipLineManager`

---

## Surface Scanning Sub-Pipeline

An additional feature for analyzing labeled flat surfaces (e.g., product labels, whiteboards, control panels).

```
User draws surface (4 corners)
  → Capture image of that region
  → OCR: extract raw text + word-level bounding boxes
  → Semantic analysis: identify functional regions (not just text)
  → Create 3D labels on surface for each semantic region
  → Generate contextual questions about the content
  → User taps label → AI explains that region
  → User taps question → AI answers in context
```

**Steps in detail:**

1. **Surface Drawing (`DragSurface`)**: State machine — user pinches 4 corners sequentially to define a rectangular region in 3D space. Emits `OnSurfaceCompleted` event with 4 world-space corner vectors.

2. **Image Capture & OCR (`CloudVisionOCRUnified`)**: Crops the render texture to the surface region; sends base64-encoded PNG to Google Cloud Vision API. Returns full text string + per-word bounding boxes in normalized image coordinates.

3. **Semantic Region Extraction (`SurfaceScanOCR`)**: Sends the image + OCR text to Gemini Vision with a prompt requesting **functional areas** (not raw text lines). Returns a JSON array:
   ```json
   [
     {
       "label": "Nutrition Facts Section",
       "boundingBox": { "x": 0.1, "y": 0.3, "width": 0.4, "height": 0.5 }
     }
   ]
   ```
   These semantic regions group related text and non-text elements (icons, ports, diagrams) into meaningful units.

4. **3D Label Placement**: Bounding box coordinates (normalized 0–1) are mapped to world-space positions on the surface plane using the surface's corner points and normal vector. A semantic label prefab is instantiated at each position, facing the user.

5. **Question Generation**: All semantic region labels are aggregated into a context string and sent to Gemini with a prompt requesting relevant user questions in JSON format. Each question becomes a tappable UI element in a `LazyFollow` panel.

6. **Interactive Labels**: Tapping a semantic label sends the label context + image to Gemini and displays a detailed explanation in a floating panel anchored relative to the surface normal.

**Implementation classes:** `DragSurface`, `SurfaceScanOCR`, `CloudVisionOCRUnified`, `GeminiQuestionAnswerer`

---

## Camera / Frame Capture

All AI components require a camera frame as input. The system maintains a live RenderTexture from the passthrough camera:

- **On-device**: A native plugin (Swift/Kotlin) captures the device's physical world camera and writes pixels to a shared texture pointer. A Unity bridge (`VisionProCameraBridge`) polls this pointer and copies into a `RenderTexture`.
- **In editor / development**: Falls back to a `WebCamTexture` targeting any available webcam.
- **Frame encoding**: Components call a shared utility (in `GeminiGeneral`) that reads the RenderTexture, encodes it as PNG, and converts to a base64 string for API submission.

---

## AI Integration Architecture

All AI-using components inherit from `GeminiGeneral`, which provides:
- API key and model name initialization
- Frame capture and base64 encoding
- HTTP request construction and response parsing
- Concurrent request management (multiple components can call the API simultaneously without interference)

**API endpoint pattern:**
```
POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}
Body: { contents: [{ parts: [{ text: "..." }, { inlineData: { mimeType: "image/png", data: "..." } }] }] }
```

**Model used:** `gemini-2.0-flash` (multimodal, vision-capable)

**Components using AI:**

| Component | AI Task |
|---|---|
| `Gemini2DBoundingBoxDetector` | Object detection: → 2D bounding boxes + labels |
| `HandGrabbingDetector` | Hand state recognition: → which hand, which object |
| `SurfaceScanOCR` | Semantic region identification: → functional area JSON |
| `ObjectComparisonManager` | Comparison generation: → similarities/differences JSON |
| `GeminiQuestionAnswerer` | Question answering: → natural language response |
| `SceneContextManager` | Scene understanding: → scene type, tasks, objects |
| `SphereToggleScript` | Per-object question generation: → question list JSON |

---

## Hand Tracking & Gesture Detection

Hand joint data is read from the XR subsystem (`XRHandSubsystem`). Three distinct pinch gestures are distinguished by which finger closes to the thumb:

| Finger | Gesture | Action |
|---|---|---|
| Index | Standard pinch | Select / draw surface / confirm |
| Middle | Multi-select pinch | Activate brush selection mode |
| Ring | Menu pinch | Toggle UI visibility |

Pinch state is determined by the distance between fingertip and thumb tip falling below a configurable threshold. Events are emitted on pinch start and end.

For object grabbing, a separate AI-based detector (`HandGrabbingDetector`) analyzes camera frames to determine if a hand is holding a specific object — enabling the system to reparent anchors to follow the hand during physical manipulation.

**Implementation classes:** `MyHandTracking`, `HandGrabbingDetector`, `HandGrabTrigger`

---

## Object Registry

`SceneObjectManager` is a singleton maintaining the authoritative list of all detected objects:
- Prevents duplicate registrations using a configurable distance threshold (`matchingRadius`)
- Stores `SceneObjectAnchor` records: `{ label, position, GameObject, boundingRadius, userLocked }`
- Fires `OnAnchorCountChanged` to notify subscribers (e.g., triggers comparison when exactly 2 are selected)
- Coordinates with `RelationshipLineManager` for drawing spatial relationship lines between objects

---

## Key Architectural Patterns

- **Event-driven**: Components communicate through C# events (`OnSurfaceCompleted`, `OnPinchStarted`, `OnOCRComplete`, `OnAnchorCountChanged`) rather than direct references.
- **Inheritance for AI**: `GeminiGeneral` base class handles all API boilerplate; subclasses implement only prompt construction and response parsing.
- **Singleton registry**: `SceneObjectManager` provides global state for detected objects.
- **State machines**: Multi-step user flows (surface drawing, proxy interaction modes) use explicit enum-based states.
- **Coroutines for async**: All API calls and multi-step sequences use Unity coroutines to avoid blocking the main thread.
- **Concurrent API calls**: `GeminiAPI` uses a `ConcurrentDictionary` keyed by request ID so multiple components can call Gemini simultaneously.

---

## Application Scenarios (from Paper)

1. **Everyday Information Retrieval**: Scan an office or kitchen; skim through object attribute panels; filter/group by semantic attribute (e.g., show all XR-related books); brush-select multiple items for aggregate operations.

2. **Building Navigation**: Use predefined digital twin data instead of AI scene parsing. Pinch a building to reveal its floor structure with an X-ray effect. Zoom between floors and rooms; group rooms by department via semantic attributes.

3. **Multi-Drone Control**: Proxies represent tracked dynamic objects. Select drones by spatial position or attribute (e.g., battery level); issue movement commands through pinch-and-move gestures on proxies.

---

## Design Principles (from Expert Evaluation)

- **No new gestures**: All interactions use familiar pinch and direct-manipulation vocabulary.
- **Dual feedback**: When a proxy is selected, highlight both the proxy and the physical object simultaneously.
- **Preserve spatial topology**: Proxy layout must maintain left/right/above/below relationships from the real world.
- **Human override**: AI detection errors must be correctable; provide manual registration as fallback.
- **Complementary AI + Human**: Treat AI scene parsing as a starting point, not ground truth.

---

## File Structure

```
Assets/
├── Script/                  # Core custom scripts
│   ├── DragSurface.cs               # Surface drawing state machine
│   ├── SurfaceScanOCR.cs            # OCR + semantic labeling pipeline
│   ├── CloudVisionOCRUnified.cs     # Google Cloud Vision API client
│   ├── GeminiGeneral.cs             # AI base class
│   ├── GeminiAPI.cs                 # HTTP client for Gemini
│   ├── Gemini2DBoundingBoxDetector.cs # Object detection
│   ├── GeminiRaycast.cs             # 2D bbox → 3D position
│   ├── SceneObjectManager.cs        # Object registry (singleton)
│   ├── SceneObjectAnchor.cs         # Anchor data structure
│   ├── MyHandTracking.cs            # Hand joint + pinch detection
│   ├── HandGrabbingDetector.cs      # AI-based grab detection
│   ├── HandGrabTrigger.cs           # Anchor grab behavior
│   ├── ObjectComparisonManager.cs   # Two-object comparison
│   ├── ComparisonPanel.cs           # Comparison UI
│   ├── MultiSelectManager.cs        # Multi-select mode coordination
│   ├── RelationshipLineManager.cs   # Spatial relationship visualization
│   ├── SceneContextManager.cs       # Periodic scene analysis
│   ├── GeminiQuestionAnswerer.cs    # Q&A with context
│   └── SphereToggleScript.cs        # Per-object selection + questions
├── (root)/
│   ├── VisionProCameraBridge.cs     # Camera texture bridge
│   ├── BaselineModeController.cs    # A/B mode toggle
│   ├── FOVAdjuster.cs               # Camera FOV control
│   └── ObjectTrackingToggleConfigure.cs # Grab behavior config
├── Prefab/
│   ├── LabelObject.prefab           # 3D object label
│   ├── OCRLinePrefab.prefab         # OCR text line display
│   ├── SemanticLinePrefab.prefab    # Semantic region label
│   ├── ComparisonPanel.prefab       # Comparison results panel
│   ├── Question.prefab              # Tappable question button
│   └── Description Panel.prefab    # Object detail panel
├── ForSwift/
│   └── VisionProCameraAccess.swift  # Native camera plugin
└── docs/
    └── SurfaceScanningFeature.md    # Surface scanning documentation
```

---

## External Dependencies

| Service / Library | Role |
|---|---|
| Google Generative AI (Gemini) | Object detection, semantic analysis, comparison, Q&A |
| Google Cloud Vision API | OCR with word-level bounding boxes |
| XR Interaction Toolkit | Hand tracking, interaction primitives |
| Unity XR Hands | Raw hand joint pose data |
| PolySpatial / visionOS | Spatial rendering and input on Apple Vision Pro |
| Newtonsoft.Json | JSON parsing for AI responses |
| TextMeshPro | 3D text rendering for labels |
| Z3 (paper) | Constraint-based proxy layout optimization |
