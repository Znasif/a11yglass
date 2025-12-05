/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models

import com.google.gson.JsonElement
import com.google.gson.annotations.SerializedName

/**
 * Generic server response that can contain various types of data.
 * The server can send:
 * - image: Base64 encoded processed image
 * - text: Text response (can be string or object for point cloud)
 * - status: Status message
 * - error: Error message
 * - Audio playback chunks
 */
data class ServerResponse(
    @SerializedName("image")
    val image: String? = null,
    
    @SerializedName("text")
    val text: JsonElement? = null,  // Can be String or Object (for point cloud)
    
    @SerializedName("status")
    val status: String? = null,
    
    @SerializedName("error")
    val error: String? = null,
    
    @SerializedName("type")
    val type: String? = null,
    
    // Audio playback fields
    @SerializedName("audio_chunk")
    val audioChunk: String? = null,
    
    @SerializedName("chunk_index")
    val chunkIndex: Int? = null,
    
    @SerializedName("total_chunks")
    val totalChunks: Int? = null,
    
    @SerializedName("is_last_chunk")
    val isLastChunk: Boolean? = null,
    
    // Audio recording status fields
    @SerializedName("session_id")
    val sessionId: String? = null,
    
    @SerializedName("filepath")
    val filepath: String? = null,
    
    @SerializedName("duration")
    val duration: String? = null,
    
    @SerializedName("streaming_back")
    val streamingBack: Boolean? = null,
    
    // Processor control (server can request processor change)
    @SerializedName("processor_id")
    val processorId: Int? = null,
    
    @SerializedName("reason")
    val reason: String? = null
) {
    /**
     * Check if this response is an audio playback message.
     */
    fun isAudioPlayback(): Boolean = type == "audio_stream_playback" && audioChunk != null
    
    /**
     * Check if this response is a processor set command.
     */
    fun isSetProcessor(): Boolean = text?.isJsonPrimitive == true && 
            text.asJsonPrimitive.isString && 
            text.asString == "set_processor"
    
    /**
     * Get text as string, or null if it's not a string.
     */
    fun getTextAsString(): String? {
        return if (text?.isJsonPrimitive == true && text.asJsonPrimitive.isString) {
            text.asString
        } else {
            null
        }
    }
    
    /**
     * Check if text contains point cloud data.
     */
    fun isPointCloudData(): Boolean {
        return text?.isJsonObject == true && 
                text.asJsonObject.has("points") &&
                text.asJsonObject.get("points").isJsonArray
    }
}

/**
 * Represents the different types of responses we can receive.
 */
sealed class ParsedResponse {
    data class ImageAndText(
        val image: String?,
        val text: String?
    ) : ParsedResponse()
    
    data class AudioPlayback(
        val audioChunk: ByteArray,
        val chunkIndex: Int,
        val totalChunks: Int,
        val isLastChunk: Boolean
    ) : ParsedResponse() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as AudioPlayback
            return audioChunk.contentEquals(other.audioChunk) &&
                    chunkIndex == other.chunkIndex &&
                    totalChunks == other.totalChunks &&
                    isLastChunk == other.isLastChunk
        }
        
        override fun hashCode(): Int {
            var result = audioChunk.contentHashCode()
            result = 31 * result + chunkIndex
            result = 31 * result + totalChunks
            result = 31 * result + isLastChunk.hashCode()
            return result
        }
    }
    
    data class SetProcessor(
        val processorId: Int,
        val reason: String?
    ) : ParsedResponse()
    
    data class Status(
        val message: String
    ) : ParsedResponse()
    
    data class Error(
        val message: String
    ) : ParsedResponse()
    
    data class AudioRecordingStatus(
        val status: String,
        val sessionId: String?,
        val filepath: String?
    ) : ParsedResponse()
}
