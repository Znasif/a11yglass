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
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Manages Text-to-Speech for speaking server text responses.
 * Implements a queue system with dynamic speech rate based on queue size.
 */
class TextToSpeechManager(context: Context) {
    
    companion object {
        private const val TAG = "TextToSpeechManager"
        
        // Speech rate configuration
        private const val BASE_SPEECH_RATE = 1.75f
        private const val MAX_SPEECH_RATE = 2.5f
        private const val RATE_INCREMENT = 0.25f
        private const val MAX_QUEUE_SIZE = 2
    }
    
    private var tts: TextToSpeech? = null
    private var isInitialized = false
    
    private val _isMuted = MutableStateFlow(false)
    val isMuted: StateFlow<Boolean> = _isMuted.asStateFlow()
    
    private val _isSpeaking = MutableStateFlow(false)
    val isSpeaking: StateFlow<Boolean> = _isSpeaking.asStateFlow()
    
    private val utteranceQueue = ConcurrentLinkedQueue<String>()
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
                // Process next item in queue
                processQueue()
            }
            
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                _isSpeaking.value = false
                Log.e(TAG, "TTS error for utterance: $utteranceId")
                processQueue()
            }
            
            override fun onError(utteranceId: String?, errorCode: Int) {
                _isSpeaking.value = false
                Log.e(TAG, "TTS error $errorCode for utterance: $utteranceId")
                processQueue()
            }
        })
    }
    
    /**
     * Speak the given text.
     * Text is added to a queue and spoken sequentially.
     * Queue size is limited to avoid backlog.
     * 
     * @param text The text to speak
     */
    fun speak(text: String) {
        if (_isMuted.value || !isInitialized || text.isBlank()) {
            return
        }
        
        // Avoid duplicate text
        val lastInQueue = utteranceQueue.peek()
        if (text == lastInQueue) return
        if (utteranceQueue.isEmpty() && _isSpeaking.value && text == lastSpokenText) return
        
        // Enforce max queue size - drop oldest if full
        if (utteranceQueue.size >= MAX_QUEUE_SIZE) {
            utteranceQueue.poll()
            Log.d(TAG, "Queue full, dropped oldest utterance")
        }
        
        utteranceQueue.add(text)
        
        // Start processing if not already speaking
        if (!_isSpeaking.value) {
            processQueue()
        }
    }
    
    /**
     * Process the next item in the utterance queue.
     */
    private fun processQueue() {
        if (_isMuted.value || utteranceQueue.isEmpty()) {
            return
        }
        
        val textToSpeak = utteranceQueue.poll() ?: return
        lastSpokenText = textToSpeak
        
        // Calculate dynamic speech rate based on queue length
        val queueLength = utteranceQueue.size
        var speechRate = BASE_SPEECH_RATE + (queueLength * RATE_INCREMENT)
        speechRate = minOf(speechRate, MAX_SPEECH_RATE)
        
        tts?.setSpeechRate(speechRate)
        
        val utteranceId = "utterance_${utteranceCounter++}"
        
        Log.d(TAG, "Speaking at rate: $speechRate, items waiting: $queueLength")
        
        tts?.speak(textToSpeak, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
    }
    
    /**
     * Set mute state.
     * @param muted Whether TTS should be muted
     */
    fun setMuted(muted: Boolean) {
        _isMuted.value = muted
        
        if (muted) {
            // Clear queue and stop current speech
            utteranceQueue.clear()
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
     * Stop current speech and clear queue.
     */
    fun stop() {
        utteranceQueue.clear()
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
