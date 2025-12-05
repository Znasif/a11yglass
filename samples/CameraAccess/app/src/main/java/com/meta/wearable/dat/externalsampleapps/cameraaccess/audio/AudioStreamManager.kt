/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.*
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Manages microphone audio capture and streaming.
 * Captures PCM audio at 24kHz, mono, 16-bit (matching server expectations).
 */
class AudioStreamManager(private val context: Context) {
    
    companion object {
        private const val TAG = "AudioStreamManager"
        
        // Audio configuration matching server expectations
        const val SAMPLE_RATE = 24000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        
        // Buffer sizes
        private const val BUFFER_SIZE_FACTOR = 2
    }
    
    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    
    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()
    
    private var onAudioChunk: ((ByteArray) -> Unit)? = null
    
    // Calculate minimum buffer size
    private val minBufferSize = AudioRecord.getMinBufferSize(
        SAMPLE_RATE,
        CHANNEL_CONFIG,
        AUDIO_FORMAT
    )
    
    private val bufferSize = minBufferSize * BUFFER_SIZE_FACTOR
    
    /**
     * Check if we have recording permission.
     */
    fun hasRecordingPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    /**
     * Start recording audio from the microphone.
     * @param onChunk Callback invoked with each audio chunk (PCM data)
     * @return true if recording started successfully
     */
    fun startRecording(onChunk: (ByteArray) -> Unit): Boolean {
        if (_isRecording.value) {
            Log.w(TAG, "Already recording")
            return false
        }
        
        if (!hasRecordingPermission()) {
            Log.e(TAG, "No recording permission")
            return false
        }
        
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            Log.e(TAG, "Invalid buffer size: $minBufferSize")
            return false
        }
        
        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )
            
            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                audioRecord?.release()
                audioRecord = null
                return false
            }
            
            onAudioChunk = onChunk
            audioRecord?.startRecording()
            _isRecording.value = true
            
            // Start the recording loop
            recordingJob = scope.launch {
                recordingLoop()
            }
            
            Log.d(TAG, "Started recording at ${SAMPLE_RATE}Hz, buffer size: $bufferSize")
            return true
            
        } catch (e: SecurityException) {
            Log.e(TAG, "Security exception starting recording: ${e.message}")
            return false
        } catch (e: Exception) {
            Log.e(TAG, "Error starting recording: ${e.message}")
            audioRecord?.release()
            audioRecord = null
            return false
        }
    }
    
    /**
     * Stop recording audio.
     */
    fun stopRecording() {
        if (!_isRecording.value) {
            Log.w(TAG, "Not recording")
            return
        }
        
        _isRecording.value = false
        recordingJob?.cancel()
        recordingJob = null
        
        try {
            audioRecord?.stop()
            audioRecord?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recording: ${e.message}")
        }
        
        audioRecord = null
        onAudioChunk = null
        
        Log.d(TAG, "Stopped recording")
    }
    
    /**
     * Main recording loop that reads audio data and invokes callback.
     */
    private suspend fun recordingLoop() {
        val buffer = ByteArray(bufferSize)
        
        while (_isRecording.value && coroutineContext.isActive) {
            try {
                val bytesRead = audioRecord?.read(buffer, 0, bufferSize) ?: -1
                
                when {
                    bytesRead > 0 -> {
                        // Copy only the bytes that were read
                        val chunk = buffer.copyOf(bytesRead)
                        onAudioChunk?.invoke(chunk)
                    }
                    bytesRead == AudioRecord.ERROR_INVALID_OPERATION -> {
                        Log.e(TAG, "Invalid operation error")
                        break
                    }
                    bytesRead == AudioRecord.ERROR_BAD_VALUE -> {
                        Log.e(TAG, "Bad value error")
                        break
                    }
                    bytesRead == AudioRecord.ERROR_DEAD_OBJECT -> {
                        Log.e(TAG, "Dead object error")
                        break
                    }
                    bytesRead == AudioRecord.ERROR -> {
                        Log.e(TAG, "Generic error")
                        break
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error in recording loop: ${e.message}")
                break
            }
        }
        
        // Ensure we stop if loop exits
        withContext(Dispatchers.Main) {
            if (_isRecording.value) {
                stopRecording()
            }
        }
    }
    
    /**
     * Clean up resources.
     */
    fun cleanup() {
        stopRecording()
        scope.cancel()
    }
}
