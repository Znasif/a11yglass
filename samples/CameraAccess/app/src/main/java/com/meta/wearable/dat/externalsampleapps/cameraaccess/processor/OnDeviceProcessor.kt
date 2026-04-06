package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor

import android.content.Context
import android.graphics.Bitmap

/**
 * Interface for processors that run on-device without a server connection.
 */
interface OnDeviceProcessor {
    val id: Int
    val name: String
    val description: String

    /**
     * Process a single frame and return the result.
     * Called on a background thread (Dispatchers.Default).
     */
    suspend fun process(frame: Bitmap): OnDeviceProcessorResult

    /**
     * Initialize the processor (load models, etc.).
     */
    fun initialize(context: Context)

    /**
     * Release resources held by this processor.
     */
    fun release()
}
