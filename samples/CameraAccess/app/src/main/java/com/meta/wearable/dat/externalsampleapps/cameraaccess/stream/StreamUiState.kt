/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamUiState - DAT Camera Streaming UI State
//
// This data class manages UI state for camera streaming operations using the DAT API.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.graphics.Bitmap
import com.meta.wearable.dat.camera.types.StreamSessionState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama.SavedPanorama

enum class CaptureButtonMode {
    CAMERA,              // Default: start sweep / photo capture
    RECORDING,           // Panorama sweep in progress — shows Stop
    PANORAMA_ANALYZING,  // Stitched; hierarchy building — Explore button disabled
    PANORAMA_DONE,       // Hierarchy ready — Explore button enabled
    PROXY_ACTIVE,        // In Reality Proxy mode — press to exit
}

data class StreamUiState(
    // DAT streaming state
    val streamSessionState: StreamSessionState = StreamSessionState.STOPPED,
    val videoFrame: Bitmap? = null,
    val capturedPhoto: Bitmap? = null,
    val isShareDialogVisible: Boolean = false,
    val timerMode: TimerMode = TimerMode.UNLIMITED,
    val remainingTimeSeconds: Long? = null,
    
    // Server streaming state
    val isStreamingToServer: Boolean = false,
    val processedFrame: Bitmap? = null,  // Processed image from server
    val responseText: String = "",        // Text response from server
    
    // Audio streaming state
    val isAudioStreaming: Boolean = false,
    val isAudioMuted: Boolean = false,
    val isPlayingAudio: Boolean = false,  // Playing back Gemini audio response
    
    // Voice command state
    val voiceTranscript: String = "",      // Last 50 chars of transcript
    val isVoiceListening: Boolean = false, // Shows if voice commands are active
    
    // Camera button mode (changes per processor phase)
    val captureButtonMode: CaptureButtonMode = CaptureButtonMode.CAMERA,

    // Status
    val statusMessage: String = "",
    val errorMessage: String? = null,

    // Saved panorama picker
    val savedPanoramas: List<SavedPanorama> = emptyList(),
    val showPanoramaPicker: Boolean = false,
) {
    /**
     * Get the frame to display.
     * Prefers processedFrame whenever it is set — this keeps the stitched panorama
     * visible after isStreamingToServer is cleared (finishPanoramaCapture), and
     * falls back to the raw camera feed otherwise.
     */
    val displayFrame: Bitmap?
        get() = processedFrame ?: videoFrame
}
