/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamScreen - DAT Camera Streaming UI
//
// This composable demonstrates the main streaming UI for DAT camera functionality. It shows how to
// display live video from wearable devices, handle server streaming, and audio controls.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.activity.ComponentActivity
import androidx.activity.compose.LocalActivity
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.meta.wearable.dat.camera.types.StreamSessionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.R
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.StreamViewModel
import com.meta.wearable.dat.externalsampleapps.cameraaccess.Wearables.WearablesViewModel

@Composable
fun StreamScreen(
    wearablesViewModel: WearablesViewModel,
    modifier: Modifier = Modifier,
    streamViewModel: StreamViewModel =
        viewModel(
            factory =
                StreamViewModel.Factory(
                    application = (LocalActivity.current as ComponentActivity).application,
                    wearablesViewModel = wearablesViewModel,
                ),
        ),
) {
    val streamUiState by streamViewModel.uiState.collectAsStateWithLifecycle()
    val wearablesUiState by wearablesViewModel.uiState.collectAsStateWithLifecycle()

    LaunchedEffect(Unit) { streamViewModel.startStream() }

    Box(modifier = modifier.fillMaxSize().background(Color.Black)) {
        // Video frame display (prefer processed frame if streaming to server)
        streamUiState.displayFrame?.let { frame ->
            Image(
                bitmap = frame.asImageBitmap(),
                contentDescription = stringResource(R.string.live_stream),
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Crop,
            )
        }

        // Loading indicator when starting
        if (streamUiState.streamSessionState == StreamSessionState.STARTING) {
            CircularProgressIndicator(
                modifier = Modifier.align(Alignment.Center),
                color = Color.White
            )
        }

        // Top bar with status and response
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .systemBarsPadding()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            // Status indicators row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Server streaming status
                StatusChip(
                    label = if (streamUiState.isStreamingToServer) "Streaming" else "Local",
                    isActive = streamUiState.isStreamingToServer,
                    activeColor = AppColor.Green
                )

                // Audio streaming status
                StatusChip(
                    label = if (streamUiState.isAudioStreaming) "Mic On" else "Mic Off",
                    isActive = streamUiState.isAudioStreaming,
                    activeColor = AppColor.Green
                )

                // Mute status
                StatusChip(
                    label = if (streamUiState.isAudioMuted) "Muted" else "Unmuted",
                    isActive = !streamUiState.isAudioMuted,
                    activeColor = Color.White
                )
            }

            // Response text display
            if (streamUiState.responseText.isNotBlank()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color.Black.copy(alpha = 0.7f))
                        .padding(12.dp)
                ) {
                    Text(
                        text = streamUiState.responseText,
                        color = Color.White,
                        style = MaterialTheme.typography.bodyMedium,
                        maxLines = 3,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }

            // Error message
            streamUiState.errorMessage?.let { error ->
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(AppColor.Red.copy(alpha = 0.8f))
                        .padding(12.dp)
                ) {
                    Text(
                        text = error,
                        color = Color.White,
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
        }

        // Bottom controls
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .padding(horizontal = 16.dp)
                .navigationBarsPadding()
                .padding(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            // Processor info
            wearablesUiState.selectedProcessor?.let { processor ->
                Text(
                    text = "Processor: ${processor.name}",
                    color = Color.White.copy(alpha = 0.7f),
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.fillMaxWidth(),
                    textAlign = TextAlign.Center
                )
            }

            // Main control buttons row
            Row(
                modifier = Modifier.fillMaxWidth().height(56.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // Stop/Back button
                SwitchButton(
                    label = stringResource(R.string.stop_stream_button_title),
                    onClick = {
                        streamViewModel.stopStream()
                        wearablesViewModel.navigateToDeviceSelection()
                    },
                    isDestructive = true,
                    modifier = Modifier.weight(1f),
                )

                // Timer button
                TimerButton(
                    timerMode = streamUiState.timerMode,
                    onClick = { streamViewModel.cycleTimerMode() },
                )

                // Photo capture button
                CaptureButton(
                    onClick = { streamViewModel.capturePhoto() },
                )
            }

            // Server streaming and audio control row
            Row(
                modifier = Modifier.fillMaxWidth().height(48.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // Server streaming toggle
                Button(
                    onClick = { streamViewModel.toggleServerStreaming() },
                    modifier = Modifier.weight(1f).fillMaxHeight(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (streamUiState.isStreamingToServer) {
                            AppColor.Red
                        } else {
                            AppColor.Green
                        }
                    )
                ) {
                    Icon(
                        imageVector = if (streamUiState.isStreamingToServer) {
                            Icons.Default.CloudOff
                        } else {
                            Icons.Default.CloudUpload
                        },
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (streamUiState.isStreamingToServer) "Stop Server" else "Start Server",
                        maxLines = 1
                    )
                }

                // Audio streaming toggle
                Button(
                    onClick = { streamViewModel.toggleAudioStreaming() },
                    modifier = Modifier.weight(1f).fillMaxHeight(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (streamUiState.isAudioStreaming) {
                            AppColor.Red
                        } else {
                            AppColor.DeepBlue
                        }
                    )
                ) {
                    Icon(
                        imageVector = if (streamUiState.isAudioStreaming) {
                            Icons.Default.MicOff
                        } else {
                            Icons.Default.Mic
                        },
                        contentDescription = null,
                        modifier = Modifier.size(20.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (streamUiState.isAudioStreaming) "Stop Audio" else "Start Audio",
                        maxLines = 1
                    )
                }

                // Mute TTS toggle
                IconButton(
                    onClick = { streamViewModel.toggleMute() },
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            if (streamUiState.isAudioMuted) Color.Gray else AppColor.DeepBlue,
                            RoundedCornerShape(8.dp)
                        )
                ) {
                    Icon(
                        imageVector = if (streamUiState.isAudioMuted) {
                            Icons.Default.VolumeOff
                        } else {
                            Icons.Default.VolumeUp
                        },
                        contentDescription = if (streamUiState.isAudioMuted) "Unmute" else "Mute",
                        tint = Color.White
                    )
                }
            }
        }

        // Countdown timer display
        streamUiState.remainingTimeSeconds?.let { seconds ->
            val minutes = seconds / 60
            val remainingSeconds = seconds % 60
            Text(
                text = stringResource(id = R.string.time_remaining, minutes, remainingSeconds),
                color = Color.White,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .navigationBarsPadding()
                    .padding(bottom = 180.dp),
                textAlign = TextAlign.Center,
            )
        }
    }

    // Photo share dialog
    streamUiState.capturedPhoto?.let { photo ->
        if (streamUiState.isShareDialogVisible) {
            SharePhotoDialog(
                photo = photo,
                onDismiss = { streamViewModel.hideShareDialog() },
                onShare = { bitmap ->
                    streamViewModel.sharePhoto(bitmap)
                    streamViewModel.hideShareDialog()
                },
            )
        }
    }
}

@Composable
private fun StatusChip(
    label: String,
    isActive: Boolean,
    activeColor: Color,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(
                if (isActive) activeColor.copy(alpha = 0.3f) 
                else Color.Gray.copy(alpha = 0.3f)
            )
            .padding(horizontal = 12.dp, vertical = 4.dp)
    ) {
        Text(
            text = label,
            color = if (isActive) activeColor else Color.Gray,
            style = MaterialTheme.typography.labelSmall
        )
    }
}
