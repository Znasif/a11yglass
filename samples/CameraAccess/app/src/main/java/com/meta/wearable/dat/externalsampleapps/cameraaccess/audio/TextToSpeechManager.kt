/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.content.Context
import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.*

/**
 * Manages Text-to-Speech for speaking server text responses.
 * Uses "latest wins" pattern - new text cancels current speech immediately.
 * Routes audio output to Bluetooth SCO (Meta glasses) when available.
 */
class TextToSpeechManager(private val context: Context) {

    companion object {
        private const val TAG = "TextToSpeechManager"
        private const val SPEECH_RATE = 1.75f
    }

    private var tts: TextToSpeech? = null
    private var isInitialized = false

    private val _isMuted = MutableStateFlow(false)
    val isMuted: StateFlow<Boolean> = _isMuted.asStateFlow()

    private val _isSpeaking = MutableStateFlow(false)
    val isSpeaking: StateFlow<Boolean> = _isSpeaking.asStateFlow()

    private var lastSpokenText = ""
    private var utteranceCounter = 0

    init {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e(TAG, "Language not supported")
                } else {
                    isInitialized = true
                    tts?.setSpeechRate(SPEECH_RATE)
                    
                    // Route audio to Bluetooth for glasses
                    routeAudioToBluetooth()
                    
                    Log.d(TAG, "TTS initialized successfully")
                    setupUtteranceListener()
                }
            } else {
                Log.e(TAG, "TTS initialization failed: $status")
            }
        }
    }

    /**
     * Route audio output to Bluetooth SCO device (Meta glasses).
     */
    private fun routeAudioToBluetooth() {
        try {
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            
            // Set audio mode for voice communication (uses SCO)
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            
            // Find and set Bluetooth SCO device
            val devices = audioManager.availableCommunicationDevices
            val btDevice = devices.find { it.type == AudioDeviceInfo.TYPE_BLUETOOTH_SCO }
            if (btDevice != null) {
                audioManager.setCommunicationDevice(btDevice)
                Log.d(TAG, "TTS audio routed to Bluetooth SCO: ${btDevice.productName}")
            } else {
                Log.d(TAG, "No Bluetooth SCO device found, using default speaker")
            }
            
            // Set TTS audio attributes to use voice communication stream
            val audioAttributes = android.media.AudioAttributes.Builder()
                .setUsage(android.media.AudioAttributes.USAGE_VOICE_COMMUNICATION)
                .setContentType(android.media.AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
            tts?.setAudioAttributes(audioAttributes)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error routing audio to Bluetooth: ${e.message}")
        }
    }

    private fun setupUtteranceListener() {
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                _isSpeaking.value = true
            }

            override fun onDone(utteranceId: String?) {
                _isSpeaking.value = false
                
                // Check for pending text and speak it if available
                pendingText?.let { text ->
                    // Run on main thread helper if needed, but TTS is thread safe.
                    // However, we want to update state safely.
                    // Recursive call to speak() will handle the check and call speakInternal
                    // BUT inside onDone we are in a binder thread.
                    // Safer to call speakInternal directly as we know we are 'done'.
                    // But to respect the flow, we just call speakInternal.
                    speakInternal(text)
                }
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                _isSpeaking.value = false
                Log.e(TAG, "TTS error for utterance: $utteranceId")
            }

            override fun onError(utteranceId: String?, errorCode: Int) {
                _isSpeaking.value = false
                Log.e(TAG, "TTS error $errorCode for utterance: $utteranceId")
            }
        })
    }


    // Buffer for the latest text that arrived while speaking
    private var pendingText: String? = null

    /**
     * Speak the given text.
     * If speaking: queues this text (overwriting any previous pending text).
     * If not speaking: speaks immediately.
     */
    fun speak(text: String) {
        if (_isMuted.value || !isInitialized || text.isBlank()) {
            return
        }

        // Skip duplicate consecutive text
        if (text == lastSpokenText && _isSpeaking.value) {
            return
        }

        // If currently speaking, queue this as the ONE pending item (conflation)
        if (_isSpeaking.value) {
            pendingText = text
            Log.d(TAG, "Speaking busy, queued pending text: ${text.take(50)}...")
            return
        }

        speakInternal(text)
    }

    private fun speakInternal(text: String) {
        lastSpokenText = text
        val utteranceId = "utterance_${utteranceCounter++}"
        pendingText = null // Clear pending as we are processing it (or a newer one)

        // QUEUE_FLUSH usage is now safe/intended because we only call this when NOT speaking,
        // OR when we explicitly want to start the next pending item.
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
        
        Log.d(TAG, "Speaking: ${text.take(50)}...")
    }

    /**
     * Set mute state.
     * @param muted Whether TTS should be muted
     */
    fun setMuted(muted: Boolean) {
        _isMuted.value = muted

        if (muted) {
            tts?.stop()
            _isSpeaking.value = false
        }

        Log.d(TAG, "TTS ${if (muted) "muted" else "unmuted"}")
    }

    /**
     * Toggle mute state.
     * @return New mute state
     */
    fun toggleMute(): Boolean {
        setMuted(!_isMuted.value)
        return _isMuted.value
    }

    /**
     * Stop current speech.
     */
    fun stop() {
        tts?.stop()
        _isSpeaking.value = false
    }

    /**
     * Clean up TTS resources.
     */
    fun cleanup() {
        stop()
        tts?.shutdown()
        tts = null
        isInitialized = false
    }
}