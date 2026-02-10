/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.content.Context
import android.content.Intent
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Manages continuous voice command recognition from Bluetooth HFP audio.
 * Scans transcribed text for processor commands like "start object detection".
 */
class VoiceCommandManager(private val context: Context) : RecognitionListener {

    companion object {
        private const val TAG = "VoiceCommandMgr"
        private const val MAX_BUFFER_LENGTH = 500 // Rolling buffer size in chars
    }

    sealed class VoiceCommand {
        data class StartProcessor(val processorId: Int, val processorName: String) : VoiceCommand()
        object StopProcessing : VoiceCommand()
        object TakePhoto : VoiceCommand()  // For VizLens: triggers OCR re-scan
        object Track : VoiceCommand()       // For VideoSegmentation: triggers object tracking
        object StopTracking : VoiceCommand() // For VideoSegmentation: stops tracking
    }

    private var speechRecognizer: SpeechRecognizer? = null
    private var recognizerIntent: Intent? = null
    private var isListening = false
    private var onCommandDetected: ((VoiceCommand) -> Unit)? = null

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    // Rolling transcript buffer
    private val transcriptBuffer = StringBuilder()
    
    private val _transcript = MutableStateFlow("")
    val transcript: StateFlow<String> = _transcript.asStateFlow()

    private val _isActive = MutableStateFlow(false)
    val isActive: StateFlow<Boolean> = _isActive.asStateFlow()

    // Processor name mappings for fuzzy matching
    private var processorMappings: List<Pair<String, Int>> = emptyList()

    /**
     * Start voice command recognition.
     */
    fun start(onCommand: (VoiceCommand) -> Unit) {
        if (isListening) {
            Log.w(TAG, "Already listening")
            return
        }

        onCommandDetected = onCommand
        
        // Build processor name mappings
        buildProcessorMappings()
        
        // Route audio to Bluetooth SCO if available
        routeAudioToBluetooth()

        // Initialize SpeechRecognizer
        if (!SpeechRecognizer.isRecognitionAvailable(context)) {
            Log.e(TAG, "Speech recognition not available")
            return
        }

        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context).apply {
            setRecognitionListener(this@VoiceCommandManager)
        }

        recognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
        }

        isListening = true
        _isActive.value = true
        transcriptBuffer.clear()
        
        startListening()
        Log.d(TAG, "Voice command recognition started")
    }

    /**
     * Stop voice command recognition.
     */
    fun stop() {
        if (!isListening) return

        isListening = false
        _isActive.value = false

        try {
            speechRecognizer?.stopListening()
            speechRecognizer?.destroy()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping speech recognizer: ${e.message}")
        }

        speechRecognizer = null
        onCommandDetected = null
        
        // Clear Bluetooth audio routing
        clearAudioRouting()
        
        Log.d(TAG, "Voice command recognition stopped")
    }

    private fun startListening() {
        if (!isListening || speechRecognizer == null) return
        
        try {
            speechRecognizer?.startListening(recognizerIntent)
        } catch (e: Exception) {
            Log.e(TAG, "Error starting listening: ${e.message}")
        }
    }

    private fun routeAudioToBluetooth() {
        try {
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            val devices = audioManager.availableCommunicationDevices

            val btDevice = devices.find { it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO }
            if (btDevice != null) {
                audioManager.setCommunicationDevice(btDevice)
                Log.d(TAG, "Audio routed to Bluetooth SCO: ${btDevice.productName}")
            } else {
                Log.d(TAG, "No Bluetooth SCO device found, using default mic")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error routing audio to Bluetooth: ${e.message}")
        }
    }

    private fun clearAudioRouting() {
        try {
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            audioManager.clearCommunicationDevice()
        } catch (e: Exception) {
            Log.e(TAG, "Error clearing audio routing: ${e.message}")
        }
    }

    private fun buildProcessorMappings() {
        processorMappings = OnDeviceProcessorManager.getOnDeviceProcessorInfoList().map { processor ->
            // Create lowercase, simplified name for matching
            val simpleName = processor.name
                .lowercase()
                .replace("(on-device)", "")
                .replace("(", "")
                .replace(")", "")
                .trim()
            simpleName to processor.id
        }
        Log.d(TAG, "Built processor mappings: $processorMappings")
    }

    private fun processTranscript(text: String) {
        // Append to rolling buffer
        transcriptBuffer.append(" ").append(text)
        
        // Trim buffer if too long
        if (transcriptBuffer.length > MAX_BUFFER_LENGTH) {
            transcriptBuffer.delete(0, transcriptBuffer.length - MAX_BUFFER_LENGTH)
        }

        val bufferText = transcriptBuffer.toString().lowercase()
        _transcript.value = bufferText.takeLast(50)

        // Scan for commands
        scanForCommands(bufferText)
    }

    private fun scanForCommands(text: String) {
        // Look for "start <processor>" pattern
        val startIndex = text.lastIndexOf("start ")
        if (startIndex >= 0) {
            val afterStart = text.substring(startIndex + 6)
            
            // Try to match a processor name
            for ((name, id) in processorMappings) {
                if (afterStart.contains(name) || 
                    afterStart.replace(" ", "").contains(name.replace(" ", ""))) {
                    Log.d(TAG, "Detected command: start $name (id=$id)")
                    
                    // Clear the buffer after detecting command
                    transcriptBuffer.clear()
                    _transcript.value = ""
                    
                    scope.launch {
                        onCommandDetected?.invoke(VoiceCommand.StartProcessor(id, name))
                    }
                    return
                }
            }
        }

        // Look for "stop processing" or "stop"
        if (text.contains("stop processing") || text.endsWith("stop ") || text.endsWith("stop")) {
            val stopIndex = text.lastIndexOf("stop")
            // Only trigger if "stop" is near the end (last 20 chars)
            if (text.length - stopIndex < 20) {
                Log.d(TAG, "Detected command: stop processing")
                
                transcriptBuffer.clear()
                _transcript.value = ""
                
                scope.launch {
                    onCommandDetected?.invoke(VoiceCommand.StopProcessing)
                }
                return
            }
        }
        
        // Look for "photo", "scan", "rescan" commands (for VizLens OCR re-scan)
        val photoKeywords = listOf("take photo", "photo", "scan", "rescan", "re-scan")
        for (keyword in photoKeywords) {
            val keywordIndex = text.lastIndexOf(keyword)
            if (keywordIndex >= 0 && text.length - keywordIndex < 25) {
                Log.d(TAG, "Detected command: $keyword (photo/scan)")

                transcriptBuffer.clear()
                _transcript.value = ""

                scope.launch {
                    onCommandDetected?.invoke(VoiceCommand.TakePhoto)
                }
                return
            }
        }

        // Look for "stop tracking" (must check before "track" to avoid false match)
        val stopTrackKeywords = listOf("stop tracking", "stop track")
        for (keyword in stopTrackKeywords) {
            val keywordIndex = text.lastIndexOf(keyword)
            if (keywordIndex >= 0 && text.length - keywordIndex < 25) {
                Log.d(TAG, "Detected command: $keyword (stop tracking)")

                transcriptBuffer.clear()
                _transcript.value = ""

                scope.launch {
                    onCommandDetected?.invoke(VoiceCommand.StopTracking)
                }
                return
            }
        }

        // Look for "track" / "track this" commands (for VideoSegmentation)
        val trackKeywords = listOf("track this", "track that", "track it", "track")
        for (keyword in trackKeywords) {
            val keywordIndex = text.lastIndexOf(keyword)
            if (keywordIndex >= 0 && text.length - keywordIndex < 25) {
                Log.d(TAG, "Detected command: $keyword (track)")

                transcriptBuffer.clear()
                _transcript.value = ""

                scope.launch {
                    onCommandDetected?.invoke(VoiceCommand.Track)
                }
                return
            }
        }
    }

    // RecognitionListener callbacks

    override fun onReadyForSpeech(params: Bundle?) {
        Log.d(TAG, "Ready for speech")
    }

    override fun onBeginningOfSpeech() {}

    override fun onRmsChanged(rmsdB: Float) {}

    override fun onBufferReceived(buffer: ByteArray?) {}

    override fun onEndOfSpeech() {
        Log.d(TAG, "End of speech, restarting...")
    }

    override fun onError(error: Int) {
        val errorMsg = when (error) {
            SpeechRecognizer.ERROR_AUDIO -> "Audio error"
            SpeechRecognizer.ERROR_CLIENT -> "Client error"
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Insufficient permissions"
            SpeechRecognizer.ERROR_NETWORK -> "Network error"
            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
            SpeechRecognizer.ERROR_NO_MATCH -> "No match"
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer busy"
            SpeechRecognizer.ERROR_SERVER -> "Server error"
            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Speech timeout"
            else -> "Unknown error $error"
        }
        Log.d(TAG, "Recognition error: $errorMsg")
        
        // Auto-restart on most errors for continuous recognition
        if (isListening && error != SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS) {
            scope.launch {
                kotlinx.coroutines.delay(500)
                startListening()
            }
        }
    }

    override fun onResults(results: Bundle?) {
        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        if (!matches.isNullOrEmpty()) {
            val text = matches[0]
            Log.d(TAG, "Final result: $text")
            processTranscript(text)
        }

        // Auto-restart for continuous recognition
        if (isListening) {
            scope.launch {
                kotlinx.coroutines.delay(100)
                startListening()
            }
        }
    }

    override fun onPartialResults(partialResults: Bundle?) {
        val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        if (!matches.isNullOrEmpty()) {
            val text = matches[0]
            Log.d(TAG, "Partial: $text")
            // Update transcript display but don't process commands on partial
            _transcript.value = text.takeLast(50)
        }
    }

    override fun onEvent(eventType: Int, params: Bundle?) {}

    fun cleanup() {
        stop()
    }
}
