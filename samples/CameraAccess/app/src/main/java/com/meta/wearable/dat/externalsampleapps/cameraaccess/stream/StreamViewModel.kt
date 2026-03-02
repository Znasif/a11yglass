/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamViewModel - DAT Camera Streaming API Demo
//
// This ViewModel demonstrates the DAT Camera Streaming APIs for:
// - Creating and managing stream sessions with wearable devices
// - Receiving video frames from device cameras
// - Capturing photos during streaming sessions
// - Sending frames to processing server
// - Audio streaming to server and playback

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.app.Application
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Base64
import android.util.Log
import androidx.core.content.FileProvider
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.meta.wearable.dat.camera.StreamSession
import com.meta.wearable.dat.camera.startStreamSession
import com.meta.wearable.dat.camera.types.PhotoData
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.StreamSessionState
import com.meta.wearable.dat.camera.types.VideoFrame
import com.meta.wearable.dat.camera.types.VideoQuality
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.AutoDeviceSelector
import com.meta.wearable.dat.core.selectors.DeviceSelector
import com.meta.wearable.dat.externalsampleapps.cameraaccess.audio.AudioPlaybackManager
import com.meta.wearable.dat.externalsampleapps.cameraaccess.audio.AudioStreamManager
import com.meta.wearable.dat.externalsampleapps.cameraaccess.audio.TextToSpeechManager
import com.meta.wearable.dat.externalsampleapps.cameraaccess.audio.VoiceCommandManager
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ParsedResponse
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorManager
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult
import com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables.WearablesViewModel
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.sample
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.Dispatchers
import androidx.annotation.OptIn
import kotlinx.coroutines.FlowPreview

class StreamViewModel(
    application: Application,
    private val wearablesViewModel: WearablesViewModel,
) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "StreamViewModel"
        private val INITIAL_STATE = StreamUiState()
        private const val JPEG_QUALITY = 30  // Match web client quality
        private const val FRAME_DELAY_MS = 100L  // Delay between frames
    }

    // AutoDeviceSelector automatically selects the first available wearable device
    private val deviceSelector: DeviceSelector = AutoDeviceSelector()
    private var streamSession: StreamSession? = null

    private val _uiState = MutableStateFlow(INITIAL_STATE)
    val uiState: StateFlow<StreamUiState> = _uiState.asStateFlow()

    private val streamTimer = StreamTimer()

    // Audio managers
    private val audioStreamManager = AudioStreamManager(application)
    private val audioPlaybackManager = AudioPlaybackManager()
    private val ttsManager = TextToSpeechManager(application)
    private val voiceCommandManager = VoiceCommandManager(application)

    // Jobs for various coroutines
    private var videoJob: Job? = null
    private var stateJob: Job? = null
    private var timerJob: Job? = null
    private var serverResponseJob: Job? = null
    private var frameStreamingJob: Job? = null
    private var localProcessingJob: Job? = null

    // Frame streaming state
    // Use SharedFlow with replay=1 and DROP_OLDEST to ensure we always have the latest frame available
    // and drop intermediate frames if processing/sending is slow.
    private val _videoFrameFlow = MutableSharedFlow<Bitmap>(
        replay = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    init {
        // Collect timer state
        timerJob = viewModelScope.launch {
            launch {
                streamTimer.timerMode.collect { mode ->
                    _uiState.update { it.copy(timerMode = mode) }
                }
            }

            launch {
                streamTimer.remainingTimeSeconds.collect { seconds ->
                    _uiState.update { it.copy(remainingTimeSeconds = seconds) }
                }
            }

            launch {
                streamTimer.isTimerExpired.collect { expired ->
                    if (expired) {
                        // Stop streaming and navigate back
                        stopStream()
                        wearablesViewModel.navigateToDeviceSelection()
                    }
                }
            }
        }

        // Collect server responses
        serverResponseJob = viewModelScope.launch {
            wearablesViewModel.serverRepository.parsedResponses
                .collect { response ->
                    handleServerResponse(response)
                }
        }

        // Collect audio playback state
        viewModelScope.launch {
            audioPlaybackManager.isPlaying.collect { isPlaying ->
                _uiState.update { it.copy(isPlayingAudio = isPlaying) }
            }
        }

        // Collect TTS mute state
        viewModelScope.launch {
            ttsManager.isMuted.collect { isMuted ->
                _uiState.update { it.copy(isAudioMuted = isMuted) }
            }
        }

        // Clear processed frame when processor changes (discard old processor's frames)
        viewModelScope.launch {
            var lastProcessorId = -1
            wearablesViewModel.uiState.collect { state ->
                if (lastProcessorId != -1 && state.selectedProcessorId != lastProcessorId) {
                    // Processor changed - clear old frame, TTS will naturally update with new responses
                    _uiState.update { it.copy(processedFrame = null, responseText = "") }
                    Log.d(TAG, "Processor changed from $lastProcessorId to ${state.selectedProcessorId}, cleared old frame")
                }
                lastProcessorId = state.selectedProcessorId
            }
        }
    }

    // ========== DAT Stream Methods ==========

    fun startStream() {
        resetTimer()
        streamTimer.startTimer()
        videoJob?.cancel()
        stateJob?.cancel()

        val streamSession =
            Wearables.startStreamSession(
                getApplication(),
                deviceSelector,
                StreamConfiguration(videoQuality = VideoQuality.HIGH, 24),
            ).also { streamSession = it }

        videoJob = viewModelScope.launch {
            streamSession.videoStream.collect { handleVideoFrame(it) }
        }

        stateJob = viewModelScope.launch {
            streamSession.state.collect { currentState ->
                val prevState = _uiState.value.streamSessionState
                _uiState.update { it.copy(streamSessionState = currentState) }

                // Navigate back when state transitioned to STOPPED
                if (currentState != prevState && currentState == StreamSessionState.STOPPED) {
                    stopStream()
                    wearablesViewModel.navigateToDeviceSelection()
                }
            }
        }

    }

    fun stopStream() {
        stopServerStreaming()
        stopAudioStreaming()
        stopVoiceCommands()

        videoJob?.cancel()
        videoJob = null
        stateJob?.cancel()
        stateJob = null
        frameStreamingJob?.cancel()
        frameStreamingJob = null
        localProcessingJob?.cancel()
        localProcessingJob = null

        streamSession?.close()
        streamSession = null
        streamTimer.stopTimer()

        _uiState.update { INITIAL_STATE }
    }

    // ========== Streaming Methods (Local & Server) ==========

    /**
     * Start streaming frames for processing.
     * Routes to on-device or server processing based on selected processor.
     */
    @OptIn(FlowPreview::class)
    fun startServerStreaming() {
        if (_uiState.value.isStreamingToServer) {
            Log.w(TAG, "Already streaming")
            return
        }

        val selectedProcessorId = wearablesViewModel.uiState.value.selectedProcessorId

        if (OnDeviceProcessorManager.isOnDeviceProcessor(selectedProcessorId)) {
            startLocalProcessing(selectedProcessorId)
        } else {
            startRemoteStreaming()
        }
    }

    /**
     * Start processing frames locally using an on-device processor.
     */
    @OptIn(FlowPreview::class)
    private fun startLocalProcessing(processorId: Int) {
        val processor = OnDeviceProcessorManager.getProcessor(processorId)
        if (processor == null) {
            _uiState.update { it.copy(errorMessage = "On-device processor not found") }
            return
        }

        _uiState.update {
            it.copy(
                isStreamingToServer = true,
                statusMessage = "Processing on-device: ${processor.name}"
            )
        }

        localProcessingJob = viewModelScope.launch(Dispatchers.Default) {
            var flow: kotlinx.coroutines.flow.Flow<Bitmap> = _videoFrameFlow.asSharedFlow()

            if (FRAME_DELAY_MS > 0) {
                flow = flow.sample(FRAME_DELAY_MS)
            }

            flow.collect { bitmap ->
                if (_uiState.value.isStreamingToServer) {
                    try {
                        val result = processor.process(bitmap)
                        handleLocalProcessorResult(result)
                    } catch (e: Exception) {
                        Log.e(TAG, "Local processing error: ${e.message}")
                    } catch (e: OutOfMemoryError) {
                        Log.e(TAG, "OOM in local processor — freeing memory and stopping")
                        System.gc()
                        stopServerStreaming()
                    }
                }
            }
        }

        Log.d(TAG, "Started local processing with: ${processor.name}")
    }

    /**
     * Handle a result from an on-device processor.
     * Reuses the same UI update pattern as handleServerResponse.
     */
    private fun handleLocalProcessorResult(result: OnDeviceProcessorResult) {
        result.processedImage?.let { bitmap ->
            _uiState.update { it.copy(processedFrame = bitmap) }
        }

        result.text?.let { text ->
            if (text.isNotBlank()) {
                _uiState.update { it.copy(responseText = text) }
                wearablesViewModel.updateServerResponseText(text)

                if (!_uiState.value.isAudioMuted) {
                    ttsManager.speak(text)
                }
            }
        }

        // Panorama-specific: when process() signals "Done: N frames", tear down the job
        // without calling stopServerStreaming() so processedFrame (the stitch) is preserved.
        if (result.text?.startsWith("Panorama complete") == true) {
            val processorId = wearablesViewModel.uiState.value.selectedProcessorId
            if (OnDeviceProcessorManager.getProcessor(processorId) is
                com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama.PanoramaProcessor
            ) {
                finishPanoramaCapture()
            }
        }
    }

    /**
     * Terminates the local processing job after a panorama sweep completes.
     * Does NOT clear processedFrame — the stitched panorama stays visible.
     */
    private fun finishPanoramaCapture() {
        localProcessingJob?.cancel()
        localProcessingJob = null
        _uiState.update { current ->
            current.copy(
                isStreamingToServer = false,
                statusMessage = "Panorama ready — tap camera button to restart",
                captureButtonMode = CaptureButtonMode.PANORAMA_DONE,
            )
        }
        Log.d(TAG, "Panorama capture finished — stitch preserved in processedFrame")
    }

    /**
     * Start streaming frames to the remote server.
     */
    @OptIn(FlowPreview::class)
    private fun startRemoteStreaming() {
        if (!wearablesViewModel.serverRepository.isConnected()) {
            _uiState.update { it.copy(errorMessage = "Not connected to server") }
            return
        }

        _uiState.update {
            it.copy(
                isStreamingToServer = true,
                statusMessage = "Streaming to server..."
            )
        }

        frameStreamingJob = viewModelScope.launch(Dispatchers.Default) {
             var flow: kotlinx.coroutines.flow.Flow<Bitmap> = _videoFrameFlow.asSharedFlow()

             if (FRAME_DELAY_MS > 0) {
                 flow = flow.sample(FRAME_DELAY_MS)
             }

             flow.collect { bitmap ->
                 if (_uiState.value.isStreamingToServer) {
                     sendFrameToServer(bitmap)
                 }
             }
        }

        Log.d(TAG, "Started remote server streaming")
    }

    /**
     * Stop streaming frames to the server.
     */
    fun stopServerStreaming() {
        frameStreamingJob?.cancel()
        frameStreamingJob = null
        localProcessingJob?.cancel()
        localProcessingJob = null

        // Stop any pending TTS (matches web client behavior)
        ttsManager.stop()

        _uiState.update {
            it.copy(
                isStreamingToServer = false,
                statusMessage = "Stopped streaming",
                processedFrame = null
            )
        }

        Log.d(TAG, "Stopped streaming")
    }

    /**
     * Toggle server streaming on/off.
     */
    fun toggleServerStreaming() {
        if (_uiState.value.isStreamingToServer) {
            stopServerStreaming()
        } else {
            startServerStreaming()
        }
    }

    /**
     * Send a frame to the server for processing.
     */
    /**
     * Send a frame to the server for processing.
     * Suspends to perform compression on Default dispatcher.
     */
    private suspend fun sendFrameToServer(bitmap: Bitmap) {
        try {
            // Compress bitmap on background thread
            // This is CPU intensive so we switch to Default dispatcher (if not already there)
            val dataUrl = kotlinx.coroutines.withContext(Dispatchers.Default) {
                val outputStream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, outputStream)
                val jpegBytes = outputStream.toByteArray()

                // Create data URL (matching web client format)
                val base64 = Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
                "data:image/jpeg;base64,$base64"
            }

            // Send to server (Main thread safe via Repository/WebSocketManager)
            val processorId = wearablesViewModel.uiState.value.selectedProcessorId
            wearablesViewModel.serverRepository.sendFrame(dataUrl, processorId)

        } catch (e: Exception) {
            Log.e(TAG, "Error sending frame: ${e.message}")
        }
    }

    // ========== Audio Methods ==========

    /**
     * Start audio streaming to server.
     */
    fun startAudioStreaming() {
        if (_uiState.value.isAudioStreaming) {
            Log.w(TAG, "Already streaming audio")
            return
        }

        if (!wearablesViewModel.serverRepository.isConnected()) {
            _uiState.update { it.copy(errorMessage = "Not connected to server") }
            return
        }

        if (!audioStreamManager.hasRecordingPermission()) {
            _uiState.update { it.copy(errorMessage = "Microphone permission required") }
            return
        }

        val started = audioStreamManager.startRecording { audioChunk ->
            // Send audio chunk to server
            wearablesViewModel.serverRepository.sendAudioChunk(audioChunk)
        }

        if (started) {
            _uiState.update {
                it.copy(
                    isAudioStreaming = true,
                    statusMessage = "Audio streaming started"
                )
            }
            Log.d(TAG, "Started audio streaming")
        } else {
            _uiState.update { it.copy(errorMessage = "Failed to start audio recording") }
        }
    }

    /**
     * Stop audio streaming to server.
     */
    fun stopAudioStreaming() {
        if (!_uiState.value.isAudioStreaming) return

        audioStreamManager.stopRecording()
        wearablesViewModel.serverRepository.sendAudioStop()

        _uiState.update {
            it.copy(
                isAudioStreaming = false,
                statusMessage = "Audio streaming stopped"
            )
        }

        Log.d(TAG, "Stopped audio streaming")
    }

    /**
     * Toggle microphone on/off.
     *
     * Enabling: always starts voice-command recognition (MLKit), and additionally
     * streams PCM audio to the server if a server connection is already open.
     * Disabling: stops both voice commands and server audio streaming.
     */
    fun toggleAudioStreaming() {
        if (_uiState.value.isAudioStreaming) {
            // Stop voice commands and server audio
            stopVoiceCommands()
            audioStreamManager.stopRecording()
            if (wearablesViewModel.serverRepository.isConnected()) {
                wearablesViewModel.serverRepository.sendAudioStop()
            }
            _uiState.update { it.copy(isAudioStreaming = false, statusMessage = "Microphone off") }
            Log.d(TAG, "Microphone disabled")
        } else {
            if (!audioStreamManager.hasRecordingPermission()) {
                _uiState.update { it.copy(errorMessage = "Microphone permission required") }
                return
            }
            // Voice commands always start with the mic
            startVoiceCommands()
            // Server audio streaming only when connected
            if (wearablesViewModel.serverRepository.isConnected()) {
                audioStreamManager.startRecording { chunk ->
                    wearablesViewModel.serverRepository.sendAudioChunk(chunk)
                }
            }
            _uiState.update { it.copy(isAudioStreaming = true, statusMessage = "Microphone on") }
            Log.d(TAG, "Microphone enabled")
        }
    }

    /**
     * Toggle TTS mute state.
     */
    fun toggleMute() {
        ttsManager.toggleMute()
    }

    // ========== Server Response Handling ==========

    private fun handleServerResponse(response: ParsedResponse) {
        when (response) {
            is ParsedResponse.ImageAndText -> {
                // Handle processed image
                response.image?.let { imageData ->
                    decodeServerImage(imageData)?.let { bitmap ->
                        _uiState.update { it.copy(processedFrame = bitmap) }
                    }
                }

                // Handle text response
                response.text?.let { text ->
                    if (text.isNotBlank()) {
                        _uiState.update { it.copy(responseText = text) }
                        wearablesViewModel.updateServerResponseText(text)

                        // Only speak if not muted (matches web client pattern)
                        if (!_uiState.value.isAudioMuted) {
                            ttsManager.speak(text)
                        }
                    }
                }
            }

            is ParsedResponse.AudioPlayback -> {
                // Add audio chunk to playback manager
                audioPlaybackManager.addAudioChunk(
                    response.audioChunk,
                    response.isLastChunk
                )
            }

            is ParsedResponse.SetProcessor -> {
                // Server requested processor change
                wearablesViewModel.selectProcessor(response.processorId)
                _uiState.update {
                    it.copy(statusMessage = "Processor changed: ${response.reason ?: ""}")
                }
            }

            is ParsedResponse.Status -> {
                _uiState.update { it.copy(statusMessage = response.message) }
            }

            is ParsedResponse.Error -> {
                _uiState.update { it.copy(errorMessage = response.message) }
            }

            is ParsedResponse.AudioRecordingStatus -> {
                _uiState.update { it.copy(statusMessage = response.status) }
            }
        }
    }

    /**
     * Decode a base64 image from the server.
     */
    private fun decodeServerImage(imageData: String): Bitmap? {
        return try {
            // Remove data URL prefix if present
            val base64Data = if (imageData.contains(",")) {
                imageData.substringAfter(",")
            } else {
                imageData
            }

            val bytes = Base64.decode(base64Data, Base64.DEFAULT)
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            Log.e(TAG, "Error decoding server image: ${e.message}")
            null
        }
    }

    // ========== Photo Capture Methods ==========

    fun capturePhoto() {
        val selectedProcessorId = wearablesViewModel.uiState.value.selectedProcessorId

        // For VizLens processor, camera button triggers OCR re-scan instead of photo capture
        val processor = OnDeviceProcessorManager.getProcessor(selectedProcessorId)
        if (processor is com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens.VizLensProcessor) {
            Log.d(TAG, "VizLens: Triggering OCR re-scan")
            processor.resetScene()
            _uiState.update { it.copy(statusMessage = "Re-scanning text...") }
            return
        }

        // For Panorama processor: 4-way state machine on camera button
        if (processor is com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama.PanoramaProcessor) {
            when {
                processor.isCapturing -> {
                    // ── Stop sweep ───────────────────────────────────────────
                    Log.d(TAG, "Panorama: stopping sweep")
                    processor.stopPanorama()
                    _uiState.update { it.copy(
                        statusMessage = "Stitching panorama…",
                        captureButtonMode = CaptureButtonMode.CAMERA,
                    ) }
                    // process() handles stopRequested on the next frame and returns
                    // "Panorama complete, N frames stitched". handleLocalProcessorResult()
                    // detects that text and calls finishPanoramaCapture() to cancel the
                    // job without clearing processedFrame (the stitched result).
                }

                processor.phase == com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama.PanoramaPhase.REALITY_PROXY -> {
                    // ── Exit Reality Proxy ───────────────────────────────────
                    Log.d(TAG, "Panorama: exiting Reality Proxy")
                    processor.exitRealityProxy()
                    finishPanoramaCapture()   // cancels localProcessingJob, preserves processedFrame
                    _uiState.update { it.copy(statusMessage = "Panorama ready — tap camera to enter proxy mode") }
                }

                processor.hasStitchedResult -> {
                    // ── Enter Reality Proxy ──────────────────────────────────
                    Log.d(TAG, "Panorama: entering Reality Proxy")
                    if (!_uiState.value.isStreamingToServer) startServerStreaming()
                    processor.enterRealityProxy()
                    _uiState.update { it.copy(
                        statusMessage = "Reality Proxy — look at objects",
                        captureButtonMode = CaptureButtonMode.PROXY_ACTIVE,
                    ) }
                }

                else -> {
                    // ── Start new sweep ──────────────────────────────────────
                    Log.d(TAG, "Panorama: starting sweep")
                    // Clear the previous stitch from processedFrame BEFORE state.reset() runs,
                    // then hint GC to release the potentially 200+ MB stitch bitmap before
                    // the new session starts allocating keyframe copies.
                    _uiState.update { it.copy(processedFrame = null) }
                    System.gc()
                    if (!_uiState.value.isStreamingToServer) startServerStreaming()
                    processor.startPanorama()
                    _uiState.update { it.copy(
                        statusMessage = "Panorama sweep started — pan slowly",
                        captureButtonMode = CaptureButtonMode.RECORDING,
                    ) }
                }
            }
            return
        }

        // For VideoSegmentation processor, camera button triggers object tracking
        if (processor is com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.videosegmentation.VideoSegmentationProcessor) {
            Log.d(TAG, "VideoSegmentation: Triggering track")
            processor.requestTrack()
            _uiState.update { it.copy(statusMessage = "Tracking object...") }
            return
        }

        // For other processors, capture photo normally
        if (uiState.value.streamSessionState == StreamSessionState.STREAMING) {
            viewModelScope.launch {
                streamSession?.capturePhoto()?.onSuccess { handlePhotoData(it) }
            }
        }
    }

    fun showShareDialog() {
        _uiState.update { it.copy(isShareDialogVisible = true) }
    }

    fun hideShareDialog() {
        _uiState.update { it.copy(isShareDialogVisible = false) }
    }

    fun sharePhoto(bitmap: Bitmap) {
        val context = getApplication<Application>()
        val imagesFolder = File(context.cacheDir, "images")
        try {
            imagesFolder.mkdirs()
            val file = File(imagesFolder, "shared_image.png")
            FileOutputStream(file).use { stream ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 90, stream)
            }

            val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
            val intent = Intent(Intent.ACTION_SEND)
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
            intent.putExtra(Intent.EXTRA_STREAM, uri)
            intent.type = "image/png"
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

            val chooser = Intent.createChooser(intent, "Share Image")
            chooser.flags = Intent.FLAG_ACTIVITY_NEW_TASK
            context.startActivity(chooser)
        } catch (e: IOException) {
            Log.e(TAG, "Failed to share photo", e)
        }
    }

    // ========== Timer Methods ==========

    fun cycleTimerMode() {
        streamTimer.cycleTimerMode()
        if (_uiState.value.streamSessionState == StreamSessionState.STREAMING) {
            streamTimer.startTimer()
        }
    }

    fun resetTimer() {
        streamTimer.resetTimer()
    }

    // ========== Frame Processing Methods ==========

    private fun handleVideoFrame(videoFrame: VideoFrame) {
        // VideoFrame contains raw I420 video data in a ByteBuffer
        val buffer = videoFrame.buffer
        val dataSize = buffer.remaining()
        val byteArray = ByteArray(dataSize)

        // Save current position
        val originalPosition = buffer.position()
        buffer.get(byteArray)
        // Restore position
        buffer.position(originalPosition)

        // Convert I420 to NV21 format which is supported by Android's YuvImage
        val nv21 = convertI420toNV21(byteArray, videoFrame.width, videoFrame.height)
        val image = YuvImage(nv21, ImageFormat.NV21, videoFrame.width, videoFrame.height, null)
        val out =
            ByteArrayOutputStream().use { stream ->
                image.compressToJpeg(Rect(0, 0, videoFrame.width, videoFrame.height), 95, stream)
                stream.toByteArray()
            }

        val bitmap = BitmapFactory.decodeByteArray(out, 0, out.size)

        // Store for server streaming via Flow
        // tryEmit will succeed because we configured BUFFER_OVERFLOW_DROP_OLDEST
        _videoFrameFlow.tryEmit(bitmap)

        _uiState.update { it.copy(videoFrame = bitmap) }
    }

    // Convert I420 (YYYYYYYY:UUVV) to NV21 (YYYYYYYY:VUVU)
    private fun convertI420toNV21(input: ByteArray, width: Int, height: Int): ByteArray {
        val output = ByteArray(input.size)
        val size = width * height
        val quarter = size / 4

        input.copyInto(output, 0, 0, size) // Y is the same

        for (n in 0 until quarter) {
            output[size + n * 2] = input[size + quarter + n] // V first
            output[size + n * 2 + 1] = input[size + n] // U second
        }
        return output
    }

    private fun handlePhotoData(photo: PhotoData) {
        val capturedPhoto =
            when (photo) {
                is PhotoData.Bitmap -> photo.bitmap
                is PhotoData.HEIC -> {
                    val byteArray = ByteArray(photo.data.remaining())
                    photo.data.get(byteArray)

                    // Extract EXIF transformation matrix and apply to bitmap
                    val exifInfo = getExifInfo(byteArray)
                    val transform = getTransform(exifInfo)
                    decodeHeic(byteArray, transform)
                }
            }
        _uiState.update { it.copy(capturedPhoto = capturedPhoto, isShareDialogVisible = true) }
    }

    // HEIC Decoding with EXIF transformation
    private fun decodeHeic(heicBytes: ByteArray, transform: Matrix): Bitmap {
        val bitmap = BitmapFactory.decodeByteArray(heicBytes, 0, heicBytes.size)
        return applyTransform(bitmap, transform)
    }

    private fun getExifInfo(heicBytes: ByteArray): ExifInterface? {
        return try {
            ByteArrayInputStream(heicBytes).use { inputStream -> ExifInterface(inputStream) }
        } catch (e: IOException) {
            Log.w(TAG, "Failed to read EXIF from HEIC", e)
            null
        }
    }

    private fun getTransform(exifInfo: ExifInterface?): Matrix {
        val matrix = Matrix()

        if (exifInfo == null) {
            return matrix // Identity matrix (no transformation)
        }

        when (
            exifInfo.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL,
            )
        ) {
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> {
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_180 -> {
                matrix.postRotate(180f)
            }
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> {
                matrix.postScale(1f, -1f)
            }
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                matrix.postRotate(90f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_90 -> {
                matrix.postRotate(90f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                matrix.postRotate(270f)
                matrix.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_ROTATE_270 -> {
                matrix.postRotate(270f)
            }
            ExifInterface.ORIENTATION_NORMAL,
            ExifInterface.ORIENTATION_UNDEFINED -> {
                // No transformation needed
            }
        }

        return matrix
    }

    private fun applyTransform(bitmap: Bitmap, matrix: Matrix): Bitmap {
        if (matrix.isIdentity) {
            return bitmap
        }

        return try {
            val transformed = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (transformed != bitmap) {
                bitmap.recycle()
            }
            transformed
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "Failed to apply transformation due to memory", e)
            bitmap
        }
    }

    // ========== Voice Command Methods ==========

    private var voiceTranscriptJob: Job? = null
    private var voiceActiveJob: Job? = null

    private fun startVoiceCommands() {
        voiceCommandManager.start { command ->
            handleVoiceCommand(command)
        }

        // Collect transcript updates
        voiceTranscriptJob = viewModelScope.launch {
            voiceCommandManager.transcript.collect { transcript ->
                _uiState.update { it.copy(voiceTranscript = transcript) }
            }
        }

        // Collect active state
        voiceActiveJob = viewModelScope.launch {
            voiceCommandManager.isActive.collect { isActive ->
                _uiState.update { it.copy(isVoiceListening = isActive) }
            }
        }

        Log.d(TAG, "Voice commands started")
    }

    private fun stopVoiceCommands() {
        voiceTranscriptJob?.cancel()
        voiceTranscriptJob = null
        voiceActiveJob?.cancel()
        voiceActiveJob = null
        voiceCommandManager.stop()
        _uiState.update { it.copy(voiceTranscript = "", isVoiceListening = false) }
        Log.d(TAG, "Voice commands stopped")
    }

    private fun handleVoiceCommand(command: VoiceCommandManager.VoiceCommand) {
        when (command) {
            is VoiceCommandManager.VoiceCommand.StartProcessor -> {
                Log.d(TAG, "Voice command: start processor ${command.processorName}")
                // Select the processor
                wearablesViewModel.selectProcessor(command.processorId)
                
                // Restart processing to use the new processor
                if (_uiState.value.isStreamingToServer) {
                    stopServerStreaming()
                }
                startServerStreaming()
            }
            is VoiceCommandManager.VoiceCommand.StopProcessing -> {
                Log.d(TAG, "Voice command: stop processing")
                if (_uiState.value.isStreamingToServer) {
                    stopServerStreaming()
                }
            }
            is VoiceCommandManager.VoiceCommand.TakePhoto -> {
                Log.d(TAG, "Voice command: take photo / scan")
                // This will trigger VizLens OCR re-scan if VizLens is active,
                // or take a photo for other processors
                capturePhoto()
            }
            is VoiceCommandManager.VoiceCommand.Track -> {
                Log.d(TAG, "Voice command: track object")
                val selectedProcessorId = wearablesViewModel.uiState.value.selectedProcessorId
                val processor = OnDeviceProcessorManager.getProcessor(selectedProcessorId)
                if (processor is com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.videosegmentation.VideoSegmentationProcessor) {
                    processor.requestTrack()
                    _uiState.update { it.copy(statusMessage = "Tracking object...") }
                } else {
                    // If not on VideoSegmentation processor, treat "track" as photo/scan
                    capturePhoto()
                }
            }
            is VoiceCommandManager.VoiceCommand.StopTracking -> {
                Log.d(TAG, "Voice command: stop tracking")
                val selectedProcessorId = wearablesViewModel.uiState.value.selectedProcessorId
                val processor = OnDeviceProcessorManager.getProcessor(selectedProcessorId)
                if (processor is com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.videosegmentation.VideoSegmentationProcessor) {
                    processor.requestStopTrack()
                    _uiState.update { it.copy(statusMessage = "Tracking stopped") }
                }
            }
        }
    }

    fun clearError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    override fun onCleared() {
        super.onCleared()
        stopStream()
        stateJob?.cancel()
        timerJob?.cancel()
        serverResponseJob?.cancel()
        streamTimer.cleanup()
        audioStreamManager.cleanup()
        audioPlaybackManager.cleanup()
        ttsManager.cleanup()
        voiceCommandManager.cleanup()
    }

    class Factory(
        private val application: Application,
        private val wearablesViewModel: WearablesViewModel,
    ) : ViewModelProvider.Factory {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(StreamViewModel::class.java)) {
                @Suppress("UNCHECKED_CAST", "KotlinGenericsCast")
                return StreamViewModel(
                    application = application,
                    wearablesViewModel = wearablesViewModel,
                ) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}