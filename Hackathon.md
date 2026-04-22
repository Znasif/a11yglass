# A11yGlass: On-Device AI & Custom Servers for Meta Smart Glasses

Welcome to the **A11yGlass (Android GlassesIO)** project for the Open Source Assistive Technology Hackathon! 

## Project Overview
A11yGlass is an open-source Android application that allows Ray-Ban Meta smart glasses to be used in hands-free mode with minimal verbal instructions for the accessibility needs of users with blindness or low vision. It bypasses default Meta AI workflows, routing the glasses' live video and audio streams directly to **your own local on-device ML models** or a **smartphone-hosted multimodal server**. 

This enables voice-initiated, always-on processors that can perform complex computer vision tasks with maximum privacy, minimal latency, and full control.

### Current Features & Capabilities
*   **Direct Wearables Integration:** Full Meta Wearables DAT SDK integration to access the hardware's camera and microphone securely.
*   **Automatic Speech Recognition Integration:** Uses android.speech.SpeechRecognizer to listen for wake words and commands like "start finger count processor".
*   **On-Device Processors:** Operates powerful ML models natively on your smartphone using MLKit and MediaPipe without requiring an internet connection. Pre-built processors include:
    *   **Finger Count Processor:** Detects how many fingers are up by using mediapipe.
    *   **Object Detection Processor:** Detects objects in the image using RF-DETR segmentation models with onnx runtime.
    *   **Caption Processor:** Summarizes the scene using FastVLM-0.5B with onnx runtime.
    *   **VizLens Processor:** Detects pointing gestures, tracks bounding boxes with SuperPoint homography, and reads text aloud via TTS.
    *   **Dense Captions Processor:** Uses Florence-2 dense region captioning via Webgpur/WASM and transformers.js.
    *   **Panorama Processor:** Captures wide panorama images by stitching multiple images from the glasses' cameras. Uses Superpoint and Lightglue for stitching, Florence-2 for region proposal, and SAM for object segmentation.
*   **Remote Multimodal Streaming:** Real-time WebSockets streaming of bounding-box frame data (10 FPS JPEG) and 24kHz audio to powerful custom backends running LLaVA, Gemma, or Qwen-VL.
*   **Conversational Architecture:** Built-in Android queued TTS fallback with Gemini-style audio response playback.

### Why Contribute?
We are expanding the boundaries of accessible computer vision wearables. If you are passionate about assistive tech, on-device ML (Kotlin/Android), or cloud multimodal AI (Python/Swift), A11yGlass represents a perfect playground to build processors that seamlessly augment spatial awareness, read complex interfaces, and assist users with everyday challenges.

---

## Workshop: Contributing to A11yGlass (2 Hours)
This 2-hour crash course is designed for Kotlin, Python, and Swift developers participating in the hackathon. By the end of this session, you’ll understand the A11yGlass architecture, successfully compile the app, and write your first computer vision processor.

### Part 1: Project Setup & Architecture (30 Min)
*   **0:00 - 0:10 | Introduction to A11yGlass:**
    *   What problem does A11yGlass solve?
    *   Overview of the repository: Wearables SDK bridge, WebSocket layer, and `OnDeviceProcessorManager`.
*   **0:10 - 0:20 | Local Environment Setup:**
    *   Cloning the repository and configuring Android Studio.
    *   Overview of prerequisites (Meta View app, Developer Mode).
    *   *Note: For the hacking period, you can use the rear cameras of your smartphones for the videofeed and the microphone of your smartphone for the audiofeed.*
*   **0:20 - 0:30 | Technical Deep Dive: The Data Flow:**
    *   How frames flow from `StreamSession` -> `WearablesViewModel` -> `Processor`.
    *   Understanding the `OnDeviceProcessorResult` contract (processed images, text for TTS, and computation time metadata).

### Part 2: Building On-Device AI with Kotlin (45 Min)
*   **0:30 - 0:45 | Analyzing the VizLens Processor:**
    *   Code walkthrough of `VizLensProcessor.kt`.
    *   Understanding MLKit OCR trigger, MediaPipe hand tracking integration, and LightGlue frame homography.
*   **0:45 - 1:15 | Hands-On Challenge: Build a "Color Finder" Processor:**
    *   Guided exercise to implement the `OnDeviceProcessor` baseline class.
    *   Implement logic to analyze the center pixel color of a frame and return its English color name for the TTS engine. 
    *   Registering the new processor in `OnDeviceProcessorManager.kt`.

### Part 3: Connecting to Cloud AI Backends (30 Min)
*   **1:15 - 1:30 | WebSocket Streaming Deep Dive:**
    *   Reviewing `WebSocketManager.kt` and `ServerApiService.kt`.
    *   Sending 64-bit encoded frames and parsing the `ServerResponse` JSON configuration.
*   **1:30 - 1:45 | Hands-On Challenge: The Python RunPod Receiver:**
    *   Setting up the server-side Python template.
    *   Extending the backend processor to ingest the glasses' stream and append a custom system prompt (e.g., *"Describe any obstacles in this frame"*).

### Part 4: Triage, Hackathon Goals & Next Steps (15 Min)
*   **1:45 - 2:00 | Where to Contribute Today:**
    *   Overview of the issue tracker and high-priority GitHub issues.
    *   Suggestions for new processors (e.g., Spatial Audio Navigation, Appliance Control Processor, Face Descriptions).
    *   Q&A, team formations, and hacking kick-off!
