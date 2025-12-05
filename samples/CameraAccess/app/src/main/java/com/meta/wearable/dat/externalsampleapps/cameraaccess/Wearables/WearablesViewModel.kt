/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// WearablesViewModel - Core DAT SDK Integration
//
// This ViewModel demonstrates the core DAT API patterns for:
// - Device registration and unregistration using the DAT SDK
// - Permission management for wearable devices
// - Device discovery and state management
// - Integration with MockDeviceKit for testing
// - Server connection and processor management

package com.meta.wearable.dat.externalsampleapps.cameraaccess.Wearables

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import com.meta.wearable.dat.core.types.RegistrationState
import com.meta.wearable.dat.mockdevice.MockDeviceKit
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.ConnectionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.ServerRepository
import kotlinx.collections.immutable.toImmutableList
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class WearablesViewModel(application: Application) : AndroidViewModel(application) {
    companion object {
        private const val TAG = "WearablesViewModel"
    }

    private val _uiState = MutableStateFlow(WearablesUiState())
    val uiState: StateFlow<WearablesUiState> = _uiState.asStateFlow()

    // Server repository for network communication
    val serverRepository = ServerRepository()

    private var monitoringStarted = false

    init {
        // Start observing server connection state
        viewModelScope.launch {
            serverRepository.connectionState.collect { state ->
                _uiState.update { 
                    it.copy(
                        connectionState = state,
                        isConnectedToServer = state is ConnectionState.Connected
                    )
                }
                
                when (state) {
                    is ConnectionState.Connected -> {
                        updateStatus("Connected to server")
                    }
                    is ConnectionState.Disconnected -> {
                        updateStatus("Disconnected from server")
                    }
                    is ConnectionState.Connecting -> {
                        updateStatus("Connecting to server...")
                    }
                    is ConnectionState.Error -> {
                        setRecentError("Connection error: ${state.message}")
                    }
                }
            }
        }
    }

    fun startMonitoring() {
        if (monitoringStarted) {
            return
        }
        monitoringStarted = true

        // This allows the app to react to registration changes (registered, unregistered, etc.)
        viewModelScope.launch {
            Wearables.registrationState.collect { value ->
                val previousState = _uiState.value.registrationState
                val showGettingStartedSheet =
                    value is RegistrationState.Registered && previousState !is RegistrationState.Registered
                _uiState.update {
                    it.copy(
                        registrationState = value, 
                        isGettingStartedSheetVisible = showGettingStartedSheet
                    )
                }
            }
        }
        // This automatically updates when devices are discovered, connected, or disconnected
        viewModelScope.launch {
            Wearables.devices.collect { value ->
                val hasMockDevices = MockDeviceKit.getInstance(getApplication()).pairedDevices.isNotEmpty()
                _uiState.update {
                    it.copy(
                        devices = value.toList().toImmutableList(), 
                        hasMockDevices = hasMockDevices
                    )
                }
            }
        }
    }

    fun startRegistration() {
        Wearables.startRegistration(getApplication())
    }

    fun startUnregistration() {
        Wearables.startUnregistration(getApplication())
    }

    // ========== Server Connection Methods ==========

    /**
     * Update the server URL.
     */
    fun setServerUrl(url: String) {
        //val normalizedUrl = serverRepository.normalizeServerUrl(url)
        _uiState.update { it.copy(serverUrl = url) }
    }

    /**
     * Connect to the WebSocket server.
     */
    fun connectToServer() {
        val url = _uiState.value.serverUrl
        Log.d(TAG, "Connecting to server: $url")
        serverRepository.connectWebSocket(url)
    }

    /**
     * Disconnect from the WebSocket server.
     */
    fun disconnectFromServer() {
        serverRepository.disconnectWebSocket()
    }

    /**
     * Fetch available processors from the server.
     */
    fun fetchProcessors() {
        viewModelScope.launch {
            _uiState.update { it.copy(isFetchingProcessors = true) }
            
            val result = serverRepository.fetchProcessors(_uiState.value.serverUrl)

            result.onSuccess { processorList ->
                val immutableProcessors = processorList.toImmutableList()  // ✅ Computed outside
                val currentSelectedId = _uiState.value.selectedProcessorId
                val newSelectedId = if (currentSelectedId == 0 && processorList.isNotEmpty()) {
                    processorList.first().id
                } else {
                    currentSelectedId
                }

                _uiState.update { state ->
                    state.copy(
                        processors = immutableProcessors,  // ✅ Just pass the pre-computed value
                        isFetchingProcessors = false,
                        selectedProcessorId = newSelectedId
                    )
                }
            }.onFailure { error ->
                _uiState.update { it.copy(isFetchingProcessors = false) }
                setRecentError("Failed to fetch processors: ${error.message}")
            }
        }
    }

    /**
     * Select a processor by ID.
     */
    fun selectProcessor(processorId: Int) {
        _uiState.update { it.copy(selectedProcessorId = processorId) }
        val processor = _uiState.value.processors.find { it.id == processorId }
        updateStatus("Selected processor: ${processor?.name ?: "Unknown"}")
    }

    // ========== Navigation Methods ==========

    fun navigateToStreaming(onRequestWearablesPermission: suspend (Permission) -> PermissionStatus) {
        viewModelScope.launch {
            val permission = Permission.CAMERA // Camera permission is required for streaming
            var permissionStatus = Wearables.checkPermissionStatus(permission)
            if (permissionStatus == PermissionStatus.Granted) {
                _uiState.update { it.copy(isStreaming = true) }
                return@launch
            }
            permissionStatus = onRequestWearablesPermission(permission)
            when (permissionStatus) {
                is PermissionStatus.Error -> {
                    setRecentError("Permission request error: ${permissionStatus.error.description}")
                }
                PermissionStatus.Denied -> {
                    setRecentError("Permission denied")
                }
                PermissionStatus.Granted -> {
                    // This is a temporary fix to address a known issue that selecting 'allow once' permission
                    // may result in video freeze.
                    delay(2000)
                    _uiState.update { it.copy(isStreaming = true) }
                }
            }
        }
    }

    fun navigateToDeviceSelection() {
        _uiState.update { it.copy(isStreaming = false) }
    }

    // ========== UI State Methods ==========

    fun showDebugMenu() {
        _uiState.update { it.copy(isDebugMenuVisible = true) }
    }

    fun hideDebugMenu() {
        _uiState.update { it.copy(isDebugMenuVisible = false) }
    }

    fun clearCameraPermissionError() {
        _uiState.update { it.copy(recentError = null) }
    }

    fun setRecentError(error: String) {
        _uiState.update { it.copy(recentError = error) }
    }

    fun showGettingStartedSheet() {
        _uiState.update { it.copy(isGettingStartedSheetVisible = true) }
    }

    fun hideGettingStartedSheet() {
        _uiState.update { it.copy(isGettingStartedSheetVisible = false) }
    }

    fun updateServerResponseText(text: String) {
        _uiState.update { it.copy(serverResponseText = text) }
    }

    private fun updateStatus(message: String) {
        _uiState.update { it.copy(lastStatusMessage = message) }
    }

    override fun onCleared() {
        super.onCleared()
        serverRepository.cleanup()
    }
}
