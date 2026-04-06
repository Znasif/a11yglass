package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor

import android.graphics.Bitmap

/**
 * Result from an on-device processor.
 */
data class OnDeviceProcessorResult(
    val processedImage: Bitmap?,
    val text: String?,
    val processingTimeMs: Long = 0
)
