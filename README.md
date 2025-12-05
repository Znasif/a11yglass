# Ray-Ban Meta Glasses → Android WhatsAI

An open-source Android app that connects Ray-Ban Meta smart glasses to **your own** multimodal AI server (e.g., LLaVA/Gemma/Qwen-VL server) instead of Meta AI.

Bypasses Meta’s cloud AI and streams live video + audio directly from the glasses to your self-hosted backend via WebSocket + REST.

## Features

- Full Meta Wearables DAT SDK integration (camera stream, photo capture, device state)
- Configurable WebSocket connection to your server (ws:// or wss://)
- Processor discovery (/processors endpoint) with dropdown selector
- ~10 FPS JPEG frame streaming (base64) from glasses camera
- 24 kHz mono PCM live microphone streaming
- Real-time processed frame display from server
- Gemini-style audio response playback
- Queue-based Android TTS fallback with dynamic rate & mute control
- Connection status, streaming indicators, response overlay

## Files Added/Modified (Full Replication of Web Client)

```
gradle/libs.versions.toml          → OkHttp, Retrofit, Gson, Coroutines
app/build.gradle.kts               → Network & audio dependencies
AndroidManifest.xml                → RECORD_AUDIO + cleartext traffic
network/models/                    → ProcessorInfo, OutgoingMessages, ServerResponse
network/WebSocketManager.kt        → WebSocket handling
network/ServerApiService.kt        → Retrofit REST interface
network/ServerRepository.kt        → Unified repo
audio/AudioStreamManager.kt        → 24 kHz mic capture
audio/AudioPlaybackManager.kt      → Gemini audio playback
audio/TextToSpeechManager.kt       → Queued TTS
wearables/* & stream/*             → ViewModels + UI state updates
ui/components/                     → ServerUrlInput, ProcessorSelector, ResponseTextDisplay
ui/NonStreamScreen.kt & StreamScreen.kt → Settings + streaming UI
MainActivity.kt                    → Permission handling
```

## Quick Start

1. Clone & open in Android Studio
2. Register your Ray-Ban Meta glasses in the Meta View app
3. Build and install the APK
4. Setup server by following: https://github.com/Znasif/HackTemplate instructions. Enter your server URL (e.g. `ws://192.168.1.100:8000/ws`)
5. Tap Connect → Fetch Processors → Select processor
6. Start Streaming → Start Server (video) → Start Audio (mic)

## Requirements

- Ray-Ban Meta glasses (firmware ≥ 8.0 recommended)
- Android device with Meta View app installed and glasses paired
- Your own multimodal server exposing:
  - GET /processors
  - WebSocket /ws accepting JSON frames + binary audio

## License

MIT (same as original Meta Wearables DAT SDK) – see LICENSE

Ready for further features (e.g. photo capture trigger, haptic feedback, offline mode). Let me know what to add next!