/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.network

import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ProcessorInfo
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ParsedResponse
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ServerResponse
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

/**
 * Repository that provides a unified interface for all server communication.
 * Handles both WebSocket streaming and REST API calls.
 */
class ServerRepository {
    
    companion object {
        private const val TAG = "ServerRepository"
        private const val DEFAULT_SERVER_URL = "ws://localhost:8000/ws"
    }
    
    private val webSocketManager = WebSocketManager()
    private var apiService: ServerApiService? = null
    private var currentBaseUrl: String = ""
    
    // Expose WebSocket state
    val connectionState: StateFlow<ConnectionState> = webSocketManager.connectionState
    val incomingMessages: SharedFlow<ServerResponse> = webSocketManager.incomingMessages
    val parsedResponses: SharedFlow<ParsedResponse> = webSocketManager.parsedResponses
    
    /**
     * Connect to the WebSocket server.
     * @param wsUrl Full WebSocket URL (e.g., "ws://192.168.1.100:8000/ws")
     */
    fun connectWebSocket(wsUrl: String) {
        webSocketManager.connect(wsUrl)
        // Also set up REST API service with the same base URL
        setupApiService(wsUrl)
    }
    
    /**
     * Disconnect from the WebSocket server.
     */
    fun disconnectWebSocket() {
        webSocketManager.disconnect()
    }
    
    /**
     * Check if connected to the server.
     */
    fun isConnected(): Boolean = webSocketManager.isConnected()
    
    /**
     * Send a video frame to the server for processing.
     * @param imageBase64 Base64 encoded JPEG image
     * @param processorId The processor to use
     */
    fun sendFrame(imageBase64: String, processorId: Int) {
        webSocketManager.sendFrame(imageBase64, processorId)
    }
    
    /**
     * Send an audio chunk to the server.
     * @param audioData Raw PCM audio bytes
     */
    fun sendAudioChunk(audioData: ByteArray) {
        webSocketManager.sendAudioChunk(audioData)
    }
    
    /**
     * Signal the server to stop audio streaming.
     */
    fun sendAudioStop() {
        webSocketManager.sendAudioStop()
    }
    
    /**
     * Fetch available processors from the server.
     * @param serverUrl WebSocket URL (will be converted to HTTP)
     * @return List of processors or empty list on error
     */
    suspend fun fetchProcessors(serverUrl: String): Result<List<ProcessorInfo>> {
        return try {
            setupApiService(serverUrl)
            
            val service = apiService ?: return Result.failure(
                Exception("API service not initialized")
            )
            
            val response = service.getProcessors()
            if (response.isSuccessful) {
                val processors = response.body()?.processors ?: emptyList()
                Log.d(TAG, "Fetched ${processors.size} processors")
                Result.success(processors)
            } else {
                Log.e(TAG, "Failed to fetch processors: ${response.code()}")
                Result.failure(Exception("HTTP ${response.code()}: ${response.message()}"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error fetching processors: ${e.message}")
            Result.failure(e)
        }
    }
    
    /**
     * Set up the Retrofit API service from a WebSocket URL.
     */
    private fun setupApiService(wsUrl: String) {
        val httpUrl = convertWsToHttp(wsUrl)
        
        // Only create new service if URL changed
        if (httpUrl == currentBaseUrl && apiService != null) {
            return
        }
        
        currentBaseUrl = httpUrl
        
        val retrofit = Retrofit.Builder()
            .baseUrl(httpUrl)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
        
        apiService = retrofit.create(ServerApiService::class.java)
        Log.d(TAG, "API service created with base URL: $httpUrl")
    }
    
    /**
     * Convert WebSocket URL to HTTP URL.
     * ws://host:port/ws -> http://host:port/
     * wss://host:port/ws -> https://host:port/
     */
    private fun convertWsToHttp(wsUrl: String): String {
        return try {
            var url = wsUrl
                .replace("wss://", "https://")
                .replace("ws://", "http://")
            
            // Remove /ws path if present
            if (url.endsWith("/ws")) {
                url = url.dropLast(3)
            }
            
            // Ensure trailing slash for Retrofit
            if (!url.endsWith("/")) {
                url = "$url/"
            }
            
            url
        } catch (e: Exception) {
            Log.e(TAG, "Error converting WS URL: ${e.message}")
            "http://localhost:8000/"
        }
    }
    
    /**
     * Validate and normalize a server URL.
     * @param input User-provided URL
     * @return Normalized WebSocket URL
     */
    fun normalizeServerUrl(input: String): String {
        var url = input.trim()
        
        // Add protocol if missing
        if (!url.startsWith("ws://") && !url.startsWith("wss://")) {
            url = "ws://$url"
        }
        
        // Add /ws path if missing
        if (!url.endsWith("/ws")) {
            if (url.endsWith("/")) {
                url = "${url}ws"
            } else {
                url = "$url/ws"
            }
        }
        
        return url
    }
    
    /**
     * Clean up resources.
     */
    fun cleanup() {
        webSocketManager.cleanup()
    }
}
