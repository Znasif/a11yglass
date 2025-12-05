/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.audio

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.ByteArrayOutputStream
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * Manages playback of audio responses from the server (Gemini audio).
 * Handles PCM audio at 24kHz, mono, 16-bit format.
 */
class AudioPlaybackManager {
    
    companion object {
        private const val TAG = "AudioPlaybackManager"
        
        // Audio configuration matching server output
        const val SAMPLE_RATE = 24000
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_OUT_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    }
    
    private var audioTrack: AudioTrack? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var playbackJob: Job? = null
    
    private val _isPlaying = MutableStateFlow(false)
    val isPlaying: StateFlow<Boolean> = _isPlaying.asStateFlow()
    
    // Buffer for accumulating audio chunks before playback
    private val audioBuffer = ByteArrayOutputStream()
    private val chunkQueue = ConcurrentLinkedQueue<ByteArray>()
    private var isAccumulatingChunks = false
    
    // Minimum buffer size for AudioTrack
    private val minBufferSize = AudioTrack.getMinBufferSize(
        SAMPLE_RATE,
        CHANNEL_CONFIG,
        AUDIO_FORMAT
    )
    
    /**
     * Initialize the AudioTrack for playback.
     */
    private fun initAudioTrack(): AudioTrack? {
        if (minBufferSize == AudioTrack.ERROR || minBufferSize == AudioTrack.ERROR_BAD_VALUE) {
            Log.e(TAG, "Invalid buffer size: $minBufferSize")
            return null
        }
        
        return try {
            val attributes = AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build()
            
            val format = AudioFormat.Builder()
                .setSampleRate(SAMPLE_RATE)
                .setChannelMask(CHANNEL_CONFIG)
                .setEncoding(AUDIO_FORMAT)
                .build()
            
            AudioTrack.Builder()
                .setAudioAttributes(attributes)
                .setAudioFormat(format)
                .setBufferSizeInBytes(minBufferSize * 2)
                .setTransferMode(AudioTrack.MODE_STREAM)
                .build()
        } catch (e: Exception) {
            Log.e(TAG, "Error creating AudioTrack: ${e.message}")
            null
        }
    }
    
    /**
     * Add an audio chunk to the playback buffer.
     * Used when receiving chunked audio from the server.
     * 
     * @param chunk PCM audio data
     * @param isLastChunk Whether this is the final chunk
     */
    fun addAudioChunk(chunk: ByteArray, isLastChunk: Boolean) {
        if (!isAccumulatingChunks) {
            isAccumulatingChunks = true
            audioBuffer.reset()
        }
        
        audioBuffer.write(chunk)
        Log.d(TAG, "Added chunk: ${chunk.size} bytes, isLast: $isLastChunk")
        
        if (isLastChunk) {
            isAccumulatingChunks = false
            val completeAudio = audioBuffer.toByteArray()
            audioBuffer.reset()
            
            Log.d(TAG, "Complete audio assembled: ${completeAudio.size} bytes")
            playAudio(completeAudio)
        }
    }
    
    /**
     * Play PCM audio data immediately.
     * @param pcmData Raw PCM audio bytes (16-bit, 24kHz, mono)
     */
    fun playAudio(pcmData: ByteArray) {
        if (pcmData.isEmpty()) {
            Log.w(TAG, "Empty audio data, skipping playback")
            return
        }
        
        // Queue audio for playback
        chunkQueue.add(pcmData)
        
        // Start playback if not already playing
        if (!_isPlaying.value) {
            startPlayback()
        }
    }
    
    /**
     * Start the playback loop.
     */
    private fun startPlayback() {
        if (_isPlaying.value) return
        
        playbackJob = scope.launch {
            _isPlaying.value = true
            
            audioTrack = initAudioTrack()
            if (audioTrack == null) {
                Log.e(TAG, "Failed to initialize AudioTrack")
                _isPlaying.value = false
                return@launch
            }
            
            audioTrack?.play()
            Log.d(TAG, "Started playback")
            
            try {
                while (isActive) {
                    val chunk = chunkQueue.poll()
                    if (chunk != null) {
                        writeAudioData(chunk)
                    } else if (chunkQueue.isEmpty()) {
                        // No more audio, wait a bit then check again
                        delay(50)
                        if (chunkQueue.isEmpty()) {
                            // Still empty, stop playback
                            break
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Playback error: ${e.message}")
            } finally {
                stopPlaybackInternal()
            }
        }
    }
    
    /**
     * Write audio data to the AudioTrack.
     */
    private fun writeAudioData(data: ByteArray) {
        var offset = 0
        while (offset < data.size) {
            val bytesToWrite = minOf(minBufferSize, data.size - offset)
            val written = audioTrack?.write(data, offset, bytesToWrite) ?: -1
            
            if (written < 0) {
                Log.e(TAG, "Error writing audio: $written")
                break
            }
            
            offset += written
        }
    }
    
    /**
     * Stop playback and release resources.
     */
    fun stopPlayback() {
        playbackJob?.cancel()
        playbackJob = null
        stopPlaybackInternal()
        chunkQueue.clear()
        audioBuffer.reset()
        isAccumulatingChunks = false
    }
    
    private fun stopPlaybackInternal() {
        try {
            audioTrack?.stop()
            audioTrack?.release()
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping AudioTrack: ${e.message}")
        }
        audioTrack = null
        _isPlaying.value = false
        Log.d(TAG, "Stopped playback")
    }
    
    /**
     * Clean up all resources.
     */
    fun cleanup() {
        stopPlayback()
        scope.cancel()
    }
}
