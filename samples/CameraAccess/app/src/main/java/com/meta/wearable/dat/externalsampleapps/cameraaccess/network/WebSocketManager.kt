/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.network

import android.util.Base64
import android.util.Log
import com.google.gson.Gson
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import okhttp3.*
import java.util.concurrent.TimeUnit

/**
 * Manages WebSocket connection to the processing server.
 * Handles sending video frames and audio chunks, and receiving processed responses.
 */
class WebSocketManager {
    
    companion object {
        private const val TAG = "WebSocketManager"
        private const val NORMAL_CLOSURE_STATUS = 1000
        private const val PING_INTERVAL_SECONDS = 30L
    }
    
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val gson = Gson()
    
    private var webSocket: WebSocket? = null
    private var serverUrl: String = ""
    
    private val client = OkHttpClient.Builder()
        .pingInterval(PING_INTERVAL_SECONDS, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(60, TimeUnit.SECONDS)
        .connectTimeout(30, TimeUnit.SECONDS)
        .build()
    
    // Connection state
    private val _connectionState = MutableStateFlow<ConnectionState>(ConnectionState.Disconnected)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()
    
    // Incoming messages
    private val _incomingMessages = MutableSharedFlow<ServerResponse>(
        replay = 0,
        extraBufferCapacity = 64
    )
    val incomingMessages: SharedFlow<ServerResponse> = _incomingMessages.asSharedFlow()
    
    // Parsed responses for easier consumption
    private val _parsedResponses = MutableSharedFlow<ParsedResponse>(
        replay = 0,
        extraBufferCapacity = 64
    )
    val parsedResponses: SharedFlow<ParsedResponse> = _parsedResponses.asSharedFlow()
    
    /**
     * Connect to the WebSocket server.
     * @param url The WebSocket URL (e.g., "ws://192.168.1.100:8000/ws")
     */
    fun connect(url: String) {
        if (_connectionState.value is ConnectionState.Connected ||
            _connectionState.value is ConnectionState.Connecting) {
            Log.w(TAG, "Already connected or connecting")
            return
        }
        
        serverUrl = url
        _connectionState.value = ConnectionState.Connecting
        
        val request = Request.Builder()
            .url(url)
            .build()
        
        webSocket = client.newWebSocket(request, createWebSocketListener())
        Log.d(TAG, "Connecting to WebSocket: $url")
    }
    
    /**
     * Disconnect from the WebSocket server.
     */
    fun disconnect() {
        webSocket?.close(NORMAL_CLOSURE_STATUS, "Client disconnecting")
        webSocket = null
        _connectionState.value = ConnectionState.Disconnected
        Log.d(TAG, "Disconnected from WebSocket")
    }
    
    /**
     * Send a video frame to the server.
     * @param imageBase64 Base64 encoded JPEG image (with or without data URL prefix)
     * @param processorId The processor ID to use for processing
     */
    fun sendFrame(imageBase64: String, processorId: Int) {
        if (_connectionState.value !is ConnectionState.Connected) {
            Log.w(TAG, "Cannot send frame: not connected")
            return
        }
        
        val message = FrameMessage(
            image = imageBase64,
            processor = processorId
        )
        
        val json = gson.toJson(message)
        webSocket?.send(json)
    }
    
    /**
     * Send an audio chunk to the server.
     * @param audioData Raw PCM audio data
     */
    fun sendAudioChunk(audioData: ByteArray) {
        if (_connectionState.value !is ConnectionState.Connected) {
            Log.w(TAG, "Cannot send audio: not connected")
            return
        }
        
        val base64Audio = Base64.encodeToString(audioData, Base64.NO_WRAP)
        val message = AudioStreamMessage(audioChunk = base64Audio)
        
        val json = gson.toJson(message)
        webSocket?.send(json)
    }
    
    /**
     * Send audio stream stop message to the server.
     */
    fun sendAudioStop() {
        if (_connectionState.value !is ConnectionState.Connected) {
            Log.w(TAG, "Cannot send audio stop: not connected")
            return
        }
        
        val message = AudioStreamStopMessage()
        val json = gson.toJson(message)
        webSocket?.send(json)
        Log.d(TAG, "Sent audio stream stop")
    }
    
    /**
     * Check if connected to the server.
     */
    fun isConnected(): Boolean = _connectionState.value is ConnectionState.Connected
    
    private fun createWebSocketListener() = object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            Log.d(TAG, "WebSocket connection opened")
            _connectionState.value = ConnectionState.Connected
        }
        
        override fun onMessage(webSocket: WebSocket, text: String) {
            scope.launch {
                try {
                    val response = gson.fromJson(text, ServerResponse::class.java)
                    _incomingMessages.emit(response)
                    
                    // Parse and emit typed response
                    val parsed = parseResponse(response)
                    if (parsed != null) {
                        _parsedResponses.emit(parsed)
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error parsing message: ${e.message}")
                }
            }
        }
        
        override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
            Log.d(TAG, "WebSocket closing: $code - $reason")
            webSocket.close(NORMAL_CLOSURE_STATUS, null)
        }
        
        override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
            Log.d(TAG, "WebSocket closed: $code - $reason")
            _connectionState.value = ConnectionState.Disconnected
        }
        
        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            Log.e(TAG, "WebSocket failure: ${t.message}")
            _connectionState.value = ConnectionState.Error(t.message ?: "Unknown error")
        }
    }
    
    private fun parseResponse(response: ServerResponse): ParsedResponse? {
        return when {
            response.isAudioPlayback() -> {
                val audioBytes = Base64.decode(response.audioChunk, Base64.DEFAULT)
                ParsedResponse.AudioPlayback(
                    audioChunk = audioBytes,
                    chunkIndex = response.chunkIndex ?: 0,
                    totalChunks = response.totalChunks ?: 1,
                    isLastChunk = response.isLastChunk ?: true
                )
            }
            
            response.isSetProcessor() -> {
                ParsedResponse.SetProcessor(
                    processorId = response.processorId ?: 0,
                    reason = response.reason
                )
            }
            
            response.status?.contains("audio") == true -> {
                ParsedResponse.AudioRecordingStatus(
                    status = response.status,
                    sessionId = response.sessionId,
                    filepath = response.filepath
                )
            }
            
            response.error != null -> {
                ParsedResponse.Error(response.error)
            }
            
            response.image != null || response.text != null -> {
                ParsedResponse.ImageAndText(
                    image = response.image,
                    text = response.getTextAsString()
                )
            }
            
            response.status != null -> {
                ParsedResponse.Status(response.status)
            }
            
            else -> null
        }
    }
    
    /**
     * Clean up resources.
     */
    fun cleanup() {
        disconnect()
        client.dispatcher.executorService.shutdown()
    }
}

/**
 * Represents the WebSocket connection state.
 */
sealed class ConnectionState {
    object Disconnected : ConnectionState()
    object Connecting : ConnectionState()
    object Connected : ConnectionState()
    data class Error(val message: String) : ConnectionState()
}
