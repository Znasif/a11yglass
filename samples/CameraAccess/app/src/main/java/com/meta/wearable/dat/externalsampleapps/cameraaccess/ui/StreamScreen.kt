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

import android.graphics.Bitmap
import androidx.activity.ComponentActivity
import androidx.activity.compose.LocalActivity
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.VolumeOff
import androidx.compose.material.icons.automirrored.filled.VolumeUp
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
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.meta.wearable.dat.camera.types.StreamSessionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.R
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama.PanoramaProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.CaptureButtonMode
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.StreamUiState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.StreamViewModel
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.components.ProcessorSpinner
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.panorama.PanoramaGalleryButton
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.panorama.PanoramaOverlay
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.panorama.PanoramaPickerDialog
import com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables.WearablesUiState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.wearables.WearablesViewModel

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

    StreamScreenContent(
        streamUiState = streamUiState,
        wearablesUiState = wearablesUiState,
        showPanoramaGallery = wearablesUiState.selectedProcessorId == PanoramaProcessor.PROCESSOR_ID,
        onStopStream = {
            streamViewModel.stopStream()
            wearablesViewModel.navigateToDeviceSelection()
        },
        onProcessorSelected = { wearablesViewModel.selectProcessor(it) },
        onToggleServerStreaming = { streamViewModel.toggleServerStreaming() },
        onToggleAudioStreaming = { streamViewModel.toggleAudioStreaming() },
        onToggleMute = { streamViewModel.toggleMute() },
        onCycleTimerMode = { streamViewModel.cycleTimerMode() },
        onCapturePhoto = { streamViewModel.capturePhoto() },
        onSharePhoto = { streamViewModel.sharePhoto(it) },
        onHideShareDialog = { streamViewModel.hideShareDialog() },
        onShowPanoramaPicker = { streamViewModel.showPanoramaPicker() },
        onHidePanoramaPicker = { streamViewModel.hidePanoramaPicker() },
        onLoadSavedPanorama = { streamViewModel.loadSavedPanorama(it) },
        onDeleteSavedPanorama = { streamViewModel.deleteSavedPanorama(it) },
        onStepNode = { streamViewModel.stepNode(it) },
        modifier = modifier,
    )
}

@Composable
fun StreamScreenContent(
    streamUiState: StreamUiState,
    wearablesUiState: WearablesUiState,
    onStopStream: () -> Unit,
    onProcessorSelected: (Int) -> Unit,
    onToggleServerStreaming: () -> Unit,
    onToggleAudioStreaming: () -> Unit,
    onToggleMute: () -> Unit,
    onCycleTimerMode: () -> Unit,
    onCapturePhoto: () -> Unit,
    onSharePhoto: (Bitmap) -> Unit,
    onHideShareDialog: () -> Unit,
    onShowPanoramaPicker: () -> Unit,
    onHidePanoramaPicker: () -> Unit,
    onLoadSavedPanorama: (String) -> Unit,
    onDeleteSavedPanorama: (String) -> Unit,
    modifier: Modifier = Modifier,
    showPanoramaGallery: Boolean = false,
    onStepNode: (Int) -> Unit = {},
) {
    val isCarouselMode = streamUiState.captureButtonMode in listOf(
        CaptureButtonMode.PANORAMA_ANALYZING,
        CaptureButtonMode.PANORAMA_DONE,
        CaptureButtonMode.PROXY_ACTIVE,
    )

    Box(modifier = modifier.fillMaxSize().background(Color.Black)) {

        if (isCarouselMode) {
            // Panorama viewer — all implementation details live in PanoramaOverlay.
            PanoramaOverlay(
                streamUiState = streamUiState,
                onStepNode = onStepNode,
                onCapturePhoto = onCapturePhoto,
            )
        } else {
            // Live camera feed or in-progress processor overlay.
            streamUiState.displayFrame?.let { frame ->
                Image(
                    bitmap = frame.asImageBitmap(),
                    contentDescription = stringResource(R.string.live_stream),
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Inside,
                )
            }
        }

        // Loading indicator when starting
        if (streamUiState.streamSessionState == StreamSessionState.STARTING) {
            CircularProgressIndicator(
                modifier = Modifier.align(Alignment.Center),
                color = Color.White,
            )
        }

        // Top bar: status chips, processor selector, response text, errors
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .systemBarsPadding()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                StatusChip(
                    label = if (streamUiState.isStreamingToServer) "Streaming" else "Local",
                    isActive = streamUiState.isStreamingToServer,
                    activeColor = AppColor.Green,
                )
                val isMicActive = streamUiState.isAudioStreaming || streamUiState.isVoiceListening
                StatusChip(
                    label = if (isMicActive) "Mic On" else "Mic Off",
                    isActive = isMicActive,
                    activeColor = AppColor.Green,
                )
                StatusChip(
                    label = if (streamUiState.isAudioMuted) "Muted" else "Unmuted",
                    isActive = !streamUiState.isAudioMuted,
                    activeColor = Color.White,
                )
            }

            ProcessorSpinner(
                processors = wearablesUiState.processors,
                selectedProcessorId = wearablesUiState.selectedProcessorId,
                onProcessorSelected = onProcessorSelected,
                enabled = wearablesUiState.processors.isNotEmpty(),
            )

            if (streamUiState.responseText.isNotBlank()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(Color.Black.copy(alpha = 0.7f))
                        .padding(12.dp),
                ) {
                    Text(
                        text = streamUiState.responseText,
                        color = Color.White,
                        style = MaterialTheme.typography.bodyMedium,
                        maxLines = 3,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
            }

            streamUiState.errorMessage?.let { error ->
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(RoundedCornerShape(8.dp))
                        .background(AppColor.Red.copy(alpha = 0.8f))
                        .padding(12.dp),
                ) {
                    Text(
                        text = error,
                        color = Color.White,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }

        // Bottom controls — hidden during Reality Proxy (PanoramaOverlay renders its own exit button)
        if (streamUiState.captureButtonMode != CaptureButtonMode.PROXY_ACTIVE) Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .padding(horizontal = 16.dp)
                .navigationBarsPadding()
                .padding(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // Row 1: Stop | Timer | Capture
            Row(
                modifier = Modifier.fillMaxWidth().height(56.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SwitchButton(
                    label = stringResource(R.string.stop_stream_button_title),
                    onClick = onStopStream,
                    isDestructive = true,
                    modifier = Modifier.weight(1f),
                )
                TimerButton(timerMode = streamUiState.timerMode, onClick = onCycleTimerMode)
                CaptureButton(mode = streamUiState.captureButtonMode, onClick = onCapturePhoto)
            }

            // Row 2: Server streaming | Audio streaming | [gallery] | Mute
            Row(
                modifier = Modifier.fillMaxWidth().height(48.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Button(
                    onClick = onToggleServerStreaming,
                    modifier = Modifier.weight(1f).fillMaxHeight(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (streamUiState.isStreamingToServer) AppColor.Red else AppColor.Green,
                    ),
                ) {
                    Icon(
                        imageVector = if (streamUiState.isStreamingToServer) Icons.Default.CloudOff else Icons.Default.CloudUpload,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (streamUiState.isStreamingToServer) "Stop Processing" else "Start Processing",
                        maxLines = 1,
                    )
                }

                Button(
                    onClick = onToggleAudioStreaming,
                    modifier = Modifier.weight(1f).fillMaxHeight(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (streamUiState.isAudioStreaming) AppColor.Red else AppColor.DeepBlue,
                    ),
                ) {
                    Icon(
                        imageVector = if (streamUiState.isAudioStreaming) Icons.Default.MicOff else Icons.Default.Mic,
                        contentDescription = null,
                        modifier = Modifier.size(20.dp),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = if (streamUiState.isAudioStreaming) "Stop Audio" else "Start Audio",
                        maxLines = 1,
                    )
                }

                if (showPanoramaGallery) {
                    PanoramaGalleryButton(onClick = onShowPanoramaPicker)
                }

                IconButton(
                    onClick = onToggleMute,
                    modifier = Modifier
                        .size(48.dp)
                        .background(
                            if (streamUiState.isAudioMuted) Color.Gray else AppColor.DeepBlue,
                            RoundedCornerShape(8.dp),
                        ),
                ) {
                    Icon(
                        imageVector = if (streamUiState.isAudioMuted) Icons.AutoMirrored.Filled.VolumeOff
                                      else Icons.AutoMirrored.Filled.VolumeUp,
                        contentDescription = if (streamUiState.isAudioMuted) "Unmute" else "Mute",
                        tint = Color.White,
                    )
                }
            }
        }

        // Countdown timer
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

        // Voice transcript
        if (streamUiState.isVoiceListening || streamUiState.voiceTranscript.isNotEmpty()) {
            Text(
                text = streamUiState.voiceTranscript.ifEmpty { "🎤 listening..." },
                color = Color.Gray,
                fontSize = 8.sp,
                maxLines = 1,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .navigationBarsPadding()
                    .padding(bottom = 4.dp),
                textAlign = TextAlign.Center,
            )
        }
    }

    // Photo share dialog
    streamUiState.capturedPhoto?.let { photo ->
        if (streamUiState.isShareDialogVisible) {
            SharePhotoDialog(
                photo = photo,
                onDismiss = onHideShareDialog,
                onShare = { bitmap -> onSharePhoto(bitmap); onHideShareDialog() },
            )
        }
    }

    // Panorama picker dialog
    if (streamUiState.showPanoramaPicker) {
        PanoramaPickerDialog(
            panoramas = streamUiState.savedPanoramas,
            onSelect = onLoadSavedPanorama,
            onDelete = onDeleteSavedPanorama,
            onDismiss = onHidePanoramaPicker,
        )
    }
}

@Preview(showBackground = true)
@Composable
private fun StreamScreenPreview() {
    StreamScreenContent(
        streamUiState = StreamUiState(),
        wearablesUiState = WearablesUiState(),
        onStopStream = {},
        onProcessorSelected = {},
        onToggleServerStreaming = {},
        onToggleAudioStreaming = {},
        onToggleMute = {},
        onCycleTimerMode = {},
        onCapturePhoto = {},
        onSharePhoto = {},
        onHideShareDialog = {},
        onShowPanoramaPicker = {},
        onHidePanoramaPicker = {},
        onLoadSavedPanorama = {},
        onDeleteSavedPanorama = {},
    )
}

@Composable
private fun StatusChip(
    label: String,
    isActive: Boolean,
    activeColor: Color,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(16.dp))
            .background(
                if (isActive) activeColor.copy(alpha = 0.3f) else Color.Gray.copy(alpha = 0.3f),
            )
            .padding(horizontal = 12.dp, vertical = 4.dp),
    ) {
        Text(
            text = label,
            color = if (isActive) activeColor else Color.Gray,
            style = MaterialTheme.typography.labelSmall,
        )
    }
}
