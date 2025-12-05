/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models

import com.google.gson.annotations.SerializedName

/**
 * Outgoing message for sending video frames to the server.
 * The server expects base64-encoded JPEG image data.
 */
data class FrameMessage(
    @SerializedName("image")
    val image: String,  // Base64 encoded JPEG with data URL prefix
    
    @SerializedName("processor")
    val processor: Int
)

/**
 * Outgoing message for starting audio streaming.
 */
data class AudioStreamMessage(
    @SerializedName("type")
    val type: String = "audio_stream",
    
    @SerializedName("audio_chunk")
    val audioChunk: String  // Base64 encoded PCM audio data
)

/**
 * Outgoing message to stop audio streaming.
 */
data class AudioStreamStopMessage(
    @SerializedName("type")
    val type: String = "audio_stream_stop"
)
