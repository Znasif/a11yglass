/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.content.Context
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
 */
class TextToSpeechManager(context: Context) {

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
                    Log.d(TAG, "TTS initialized successfully")
                    setupUtteranceListener()
                }
            } else {
                Log.e(TAG, "TTS initialization failed: $status")
            }
        }
    }

    private fun setupUtteranceListener() {
        tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                _isSpeaking.value = true
            }

            override fun onDone(utteranceId: String?) {
                _isSpeaking.value = false
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

    /**
     * Speak the given text immediately.
     * Cancels any current speech - latest text wins.
     *
     * @param text The text to speak
     */
    fun speak(text: String) {
        if (_isMuted.value || !isInitialized || text.isBlank()) {
            return
        }

        // Skip duplicate consecutive text
        if (text == lastSpokenText && _isSpeaking.value) {
            return
        }

        lastSpokenText = text
        val utteranceId = "utterance_${utteranceCounter++}"

        // QUEUE_FLUSH cancels current speech and speaks new text immediately
        tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, utteranceId)

        Log.d(TAG, "Speaking (latest wins): ${text.take(50)}...")
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