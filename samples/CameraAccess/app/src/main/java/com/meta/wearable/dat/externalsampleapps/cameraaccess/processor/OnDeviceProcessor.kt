package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Base class for processors that run on-device without a server connection.
 *
 * Provides shared infrastructure that most heavy processors need:
 *  - [isProcessing] — atomic flag for frame-drop: set true while inference runs so
 *    concurrent frames are skipped rather than queued.
 *  - [processorScope] — long-lived coroutine scope (Default dispatcher) for background
 *    work launched by the processor. Cancel it in [release] if needed.
 */
abstract class OnDeviceProcessor {
    abstract val id: Int
    abstract val name: String
    abstract val description: String

    /**
     * Process a single frame and return the result.
     * Called on a background thread (Dispatchers.Default).
     */
    abstract suspend fun process(frame: Bitmap): OnDeviceProcessorResult

    /** Initialize the processor (load models, acquire resources, etc.). */
    abstract fun initialize(context: Context)

    /** Release all resources held by this processor. */
    abstract fun release()

    /**
     * Atomic frame-drop guard. Use [AtomicBoolean.compareAndSet] to claim the lock
     * before starting inference and [AtomicBoolean.set] to release it when done.
     */
    protected val isProcessing = AtomicBoolean(false)

    /**
     * Long-lived coroutine scope for processor background work (Default dispatcher).
     * Cancel this in [release] if the processor launches ongoing coroutines.
     */
    protected val processorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
}
