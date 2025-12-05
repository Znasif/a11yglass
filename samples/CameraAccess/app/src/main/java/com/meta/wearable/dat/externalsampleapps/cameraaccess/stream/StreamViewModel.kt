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
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ParsedResponse
import com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables.WearablesViewModel
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

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

    // Jobs for various coroutines
    private var videoJob: Job? = null
    private var stateJob: Job? = null
    private var timerJob: Job? = null
    private var serverResponseJob: Job? = null
    private var frameStreamingJob: Job? = null

    // Frame streaming state
    private var lastFrameBitmap: Bitmap? = null

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
                StreamConfiguration(videoQuality = VideoQuality.MEDIUM, 24),
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

        videoJob?.cancel()
        videoJob = null
        stateJob?.cancel()
        stateJob = null
        frameStreamingJob?.cancel()
        frameStreamingJob = null

        streamSession?.close()
        streamSession = null
        streamTimer.stopTimer()
        lastFrameBitmap = null

        _uiState.update { INITIAL_STATE }
    }

    // ========== Server Streaming Methods ==========

    /**
     * Start streaming frames to the server.
     */
    fun startServerStreaming() {
        if (_uiState.value.isStreamingToServer) {
            Log.w(TAG, "Already streaming to server")
            return
        }

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

        // Start the frame streaming loop
        frameStreamingJob = viewModelScope.launch {
            while (_uiState.value.isStreamingToServer) {
                lastFrameBitmap?.let { bitmap ->
                    sendFrameToServer(bitmap)
                }
                delay(FRAME_DELAY_MS)
            }
        }

        Log.d(TAG, "Started server streaming")
    }

    /**
     * Stop streaming frames to the server.
     */
    fun stopServerStreaming() {
        frameStreamingJob?.cancel()
        frameStreamingJob = null

        // Stop any pending TTS (matches web client behavior)
        ttsManager.stop()

        _uiState.update {
            it.copy(
                isStreamingToServer = false,
                statusMessage = "Stopped streaming",
                processedFrame = null
            )
        }

        Log.d(TAG, "Stopped server streaming")
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
    private fun sendFrameToServer(bitmap: Bitmap) {
        viewModelScope.launch {
            try {
                // Convert bitmap to JPEG base64
                val outputStream = ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, outputStream)
                val jpegBytes = outputStream.toByteArray()

                // Create data URL (matching web client format)
                val base64 = Base64.encodeToString(jpegBytes, Base64.NO_WRAP)
                val dataUrl = "data:image/jpeg;base64,$base64"

                // Send to server
                val processorId = wearablesViewModel.uiState.value.selectedProcessorId
                wearablesViewModel.serverRepository.sendFrame(dataUrl, processorId)

            } catch (e: Exception) {
                Log.e(TAG, "Error sending frame: ${e.message}")
            }
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
     * Toggle audio streaming on/off.
     */
    fun toggleAudioStreaming() {
        if (_uiState.value.isAudioStreaming) {
            stopAudioStreaming()
        } else {
            startAudioStreaming()
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
                image.compressToJpeg(Rect(0, 0, videoFrame.width, videoFrame.height), 50, stream)
                stream.toByteArray()
            }

        val bitmap = BitmapFactory.decodeByteArray(out, 0, out.size)

        // Store for server streaming
        lastFrameBitmap = bitmap

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