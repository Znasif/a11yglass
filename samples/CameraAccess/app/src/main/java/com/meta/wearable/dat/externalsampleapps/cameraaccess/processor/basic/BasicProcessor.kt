package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.basic

import android.content.Context
import android.graphics.Bitmap
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult

/**
 * A simple pass-through processor that returns the original frame.
 */
class BasicProcessor : OnDeviceProcessor() {
    override val id = -100
    override val name = "Basic (On-Device)"
    override val description = "Pass-through processor - returns the original image"

    override suspend fun process(frame: Bitmap): OnDeviceProcessorResult {
        return OnDeviceProcessorResult(
            processedImage = frame,
            text = null
        )
    }

    override fun initialize(context: Context) { /* No-op */ }
    override fun release() { /* No-op */ }
}
