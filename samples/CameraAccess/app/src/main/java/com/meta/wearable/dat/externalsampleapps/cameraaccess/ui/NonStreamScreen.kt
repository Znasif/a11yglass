/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// NonStreamScreen - DAT Device Selection and Setup
//
// This screen demonstrates DAT device management and pre-streaming setup. It handles device
// registration status, camera permissions, stream readiness, and server connection.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.LinkOff
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import com.meta.wearable.dat.externalsampleapps.cameraaccess.R
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.ConnectionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.components.ProcessorSelector
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.components.ServerUrlInput
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.components.StatusBar
import com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables.WearablesViewModel
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NonStreamScreen(
    viewModel: WearablesViewModel,
    onRequestWearablesPermission: suspend (Permission) -> PermissionStatus,
    modifier: Modifier = Modifier,
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val gettingStartedSheetState = rememberModalBottomSheetState(skipPartiallyExpanded = false)
    val scope = rememberCoroutineScope()
    var dropdownExpanded by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()

    MaterialTheme(colorScheme = darkColorScheme()) {
        Box(
            modifier = modifier
                .fillMaxSize()
                .background(Color.Black)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 24.dp)
                    .padding(top = 16.dp, bottom = 100.dp)
                    .verticalScroll(scrollState),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                // Top bar with disconnect button
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .systemBarsPadding(),
                    horizontalArrangement = Arrangement.End
                ) {
                    Box {
                        IconButton(onClick = { dropdownExpanded = true }) {
                            Icon(
                                imageVector = Icons.Default.LinkOff,
                                contentDescription = "Disconnect",
                                tint = Color.White,
                                modifier = Modifier.size(28.dp)
                            )
                        }

                        DropdownMenu(
                            expanded = dropdownExpanded,
                            onDismissRequest = { dropdownExpanded = false }
                        ) {
                            DropdownMenuItem(
                                text = {
                                    Text(
                                        stringResource(R.string.unregister_button_title),
                                        color = AppColor.Red
                                    )
                                },
                                onClick = {
                                    viewModel.startUnregistration()
                                    dropdownExpanded = false
                                },
                                modifier = Modifier.height(40.dp)
                            )
                        }
                    }
                }

                // App icon and title
                Icon(
                    painter = painterResource(id = R.drawable.camera_access_icon),
                    contentDescription = stringResource(R.string.camera_access_icon_description),
                    tint = Color.White,
                    modifier = Modifier.size(60.dp * LocalDensity.current.density)
                )

                Text(
                    text = stringResource(R.string.non_stream_screen_title),
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center,
                    color = Color.White
                )

                // Status bar
                StatusBar(
                    statusMessage = uiState.lastStatusMessage,
                    errorMessage = uiState.recentError,
                    modifier = Modifier.fillMaxWidth()
                )

                // Server settings card
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFF1A1A2E)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = "Server Settings",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = Color.White
                        )

                        // Server URL input
                        ServerUrlInput(
                            serverUrl = uiState.serverUrl,
                            connectionState = uiState.connectionState,
                            isFetchingProcessors = uiState.isFetchingProcessors,
                            onUrlChange = { viewModel.setServerUrl(it) },
                            onConnect = { viewModel.connectToServer() },
                            onDisconnect = { viewModel.disconnectFromServer() },
                            onFetchProcessors = { viewModel.fetchProcessors() }
                        )

                        // Processor selector
                        ProcessorSelector(
                            processors = uiState.processors,
                            selectedProcessorId = uiState.selectedProcessorId,
                            onProcessorSelected = { viewModel.selectProcessor(it) },
                            enabled = uiState.isConnectedToServer
                        )
                    }
                }

                // Device status card
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFF1A1A2E)
                    ),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "Device Status",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            color = Color.White
                        )

                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            val deviceCount = uiState.devices.size
                            val statusColor = if (deviceCount > 0) AppColor.Green else AppColor.Yellow

                            Box(
                                modifier = Modifier
                                    .size(8.dp)
                                    .background(statusColor, RoundedCornerShape(4.dp))
                            )

                            Text(
                                text = if (deviceCount > 0) {
                                    "$deviceCount device(s) connected"
                                } else {
                                    "No devices connected"
                                },
                                style = MaterialTheme.typography.bodyMedium,
                                color = Color.White.copy(alpha = 0.8f)
                            )
                        }

                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            val serverColor = if (uiState.isConnectedToServer) AppColor.Green else Color.Gray

                            Box(
                                modifier = Modifier
                                    .size(8.dp)
                                    .background(serverColor, RoundedCornerShape(4.dp))
                            )

                            Text(
                                text = when (uiState.connectionState) {
                                    is ConnectionState.Connected -> "Server connected"
                                    is ConnectionState.Connecting -> "Connecting to server..."
                                    is ConnectionState.Error -> "Server connection error"
                                    else -> "Server not connected"
                                },
                                style = MaterialTheme.typography.bodyMedium,
                                color = Color.White.copy(alpha = 0.8f)
                            )
                        }
                    }
                }

                Spacer(modifier = Modifier.weight(1f))
            }

            // Start Streaming Button
            SwitchButton(
                label = stringResource(R.string.stream_button_title),
                onClick = { viewModel.navigateToStreaming(onRequestWearablesPermission) },
                enabled = uiState.canStartStreaming,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(horizontal = 24.dp)
                    .navigationBarsPadding()
                    .padding(bottom = 16.dp)
            )

            // Getting Started Sheet
            if (uiState.isGettingStartedSheetVisible) {
                ModalBottomSheet(
                    onDismissRequest = { viewModel.hideGettingStartedSheet() },
                    sheetState = gettingStartedSheetState
                ) {
                    GettingStartedSheetContent(
                        onContinue = {
                            scope.launch {
                                gettingStartedSheetState.hide()
                                viewModel.hideGettingStartedSheet()
                            }
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun GettingStartedSheetContent(
    onContinue: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .padding(bottom = 24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(24.dp)
    ) {
        Text(
            text = stringResource(R.string.getting_started_title),
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.SemiBold,
            textAlign = TextAlign.Center
        )

        Column(
            verticalArrangement = Arrangement.spacedBy(12.dp),
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
                .padding(bottom = 16.dp)
        ) {
            TipItem(
                iconResId = R.drawable.video_icon,
                text = stringResource(R.string.getting_started_tip_permission)
            )
            TipItem(
                iconResId = R.drawable.tap_icon,
                text = stringResource(R.string.getting_started_tip_photo)
            )
            TipItem(
                iconResId = R.drawable.smart_glasses_icon,
                text = stringResource(R.string.getting_started_tip_led)
            )
        }

        SwitchButton(
            label = stringResource(R.string.getting_started_continue),
            onClick = onContinue,
            modifier = Modifier.navigationBarsPadding()
        )
    }
}

@Composable
private fun TipItem(
    iconResId: Int,
    text: String,
    modifier: Modifier = Modifier
) {
    Row(modifier = modifier.fillMaxWidth()) {
        Icon(
            painter = painterResource(id = iconResId),
            contentDescription = "Getting started tip icon",
            modifier = Modifier
                .padding(start = 4.dp, top = 4.dp)
                .width(24.dp)
        )
        Spacer(modifier = Modifier.width(10.dp))
        Text(text = text)
    }
}
