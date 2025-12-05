/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// WearablesUiState - DAT API State Management
//
// This data class aggregates DAT API state for the UI layer

package com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables

import com.meta.wearable.dat.core.types.DeviceIdentifier
import com.meta.wearable.dat.core.types.RegistrationState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.ConnectionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ProcessorInfo
import kotlinx.collections.immutable.ImmutableList
import kotlinx.collections.immutable.persistentListOf

data class WearablesUiState(
    // DAT state
    val registrationState: RegistrationState = RegistrationState.Unavailable(),
    val devices: ImmutableList<DeviceIdentifier> = persistentListOf(),
    val recentError: String? = null,
    val isStreaming: Boolean = false,
    val hasMockDevices: Boolean = false,
    val isDebugMenuVisible: Boolean = false,
    val isGettingStartedSheetVisible: Boolean = false,
    
    // Server connection state
    val serverUrl: String = "wss://w4cdrdoe55szp3-8000.proxy.runpod.net/ws",
    val connectionState: ConnectionState = ConnectionState.Disconnected,
    val isConnectedToServer: Boolean = false,
    
    // Processor state
    val processors: ImmutableList<ProcessorInfo> = persistentListOf(),
    val selectedProcessorId: Int = 0,
    val isFetchingProcessors: Boolean = false,
    
    // Response state
    val serverResponseText: String = "",
    val lastStatusMessage: String = "",
) {
    val isRegistered: Boolean = registrationState is RegistrationState.Registered || hasMockDevices
    
    val selectedProcessor: ProcessorInfo? = processors.find { it.id == selectedProcessorId }
    
    val canStartStreaming: Boolean = isRegistered && 
            devices.isNotEmpty() && 
            isConnectedToServer
}
