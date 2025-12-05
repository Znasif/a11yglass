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
    
    // Status
    val statusMessage: String = "",
    val errorMessage: String? = null,
) {
    /**
     * Get the frame to display - prefer processed frame over raw camera frame.
     */
    val displayFrame: Bitmap?
        get() = if (isStreamingToServer && processedFrame != null) {
            processedFrame
        } else {
            videoFrame
        }
}
