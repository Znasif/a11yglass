/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.CloudOff
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.ConnectionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.AppColor

@Composable
fun ServerUrlInput(
    serverUrl: String,
    connectionState: ConnectionState,
    isFetchingProcessors: Boolean,
    onUrlChange: (String) -> Unit,
    onConnect: () -> Unit,
    onDisconnect: () -> Unit,
    onFetchProcessors: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val focusManager = LocalFocusManager.current
    val isConnected = connectionState is ConnectionState.Connected
    val isConnecting = connectionState is ConnectionState.Connecting
    
    Column(
        modifier = modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        // Server URL row
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Connection status icon
            Icon(
                imageVector = if (isConnected) Icons.Default.Cloud else Icons.Default.CloudOff,
                contentDescription = if (isConnected) "Connected" else "Disconnected",
                tint = when (connectionState) {
                    is ConnectionState.Connected -> AppColor.Green
                    is ConnectionState.Connecting -> AppColor.Yellow
                    is ConnectionState.Error -> AppColor.Red
                    else -> Color.Gray
                },
                modifier = Modifier.size(24.dp)
            )
            
            // URL text field
            OutlinedTextField(
                value = serverUrl,
                onValueChange = onUrlChange,
                modifier = Modifier.weight(1f),
                label = { Text("Server URL") },
                placeholder = { Text("ws://192.168.1.100:8000/ws") },
                singleLine = true,
                enabled = !isConnected && !isConnecting,
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Uri,
                    imeAction = ImeAction.Done
                ),
                keyboardActions = KeyboardActions(
                    onDone = {
                        focusManager.clearFocus()
                        if (!isConnected) onConnect()
                    }
                ),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = AppColor.DeepBlue,
                    unfocusedBorderColor = Color.Gray
                )
            )
        }
        
        // Action buttons row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Connect/Disconnect button
            Button(
                onClick = { if (isConnected) onDisconnect() else onConnect() },
                enabled = !isConnecting,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isConnected) AppColor.Red else AppColor.Green
                ),
                modifier = Modifier.weight(1f)
            ) {
                if (isConnecting) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = Color.White,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Connecting...")
                } else {
                    Text(if (isConnected) "Disconnect" else "Connect")
                }
            }
            
            // Fetch Processors button
            Button(
                onClick = onFetchProcessors,
                enabled = isConnected && !isFetchingProcessors,
                colors = ButtonDefaults.buttonColors(
                    containerColor = AppColor.DeepBlue
                ),
                modifier = Modifier.weight(1f)
            ) {
                if (isFetchingProcessors) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = Color.White,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Icon(
                    imageVector = Icons.Default.Refresh,
                    contentDescription = null,
                    modifier = Modifier.size(16.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text("Fetch Processors")
            }
        }
        
        // Connection status text
        if (connectionState is ConnectionState.Error) {
            Text(
                text = "Error: ${connectionState.message}",
                color = AppColor.Red,
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}
