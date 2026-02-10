package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.videosegmentation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Video Object Segmentation Processor using XMem (Track Anything).
 *
 * Interaction flow:
 * 1. IDLE: User says "track" → capture first frame
 * 2. WAITING_FOR_TARGET: Auto-select center-of-frame object (or finger-pointed object)
 * 3. INITIALIZING: Generate first-frame mask, initialize XMem memory bank
 * 4. TRACKING: Run segmentation on every frame, overlay mask, announce spatial position via TTS
 * 5. LOST: Confidence too low → prompt user to re-trigger
 *
 * Voice commands: "track" / "track this" → start, "stop tracking" / "stop" → stop
 */
class VideoSegmentationProcessor : OnDeviceProcessor {
    companion object {
        private const val TAG = "VideoSegProcessor"

        // Spatial feedback intervals
        private const val SPATIAL_FEEDBACK_INTERVAL_MS = 3000L
        private const val CONFIDENCE_LOST_THRESHOLD = 0.3f
        private const val CONFIDENCE_LOW_THRESHOLD = 0.5f

        // Grace period: don't check confidence for the first N tracking frames
        private const val GRACE_PERIOD_FRAMES = 10
        // Tolerate up to N consecutive null frames before declaring LOST
        private const val MAX_CONSECUTIVE_NULL_FRAMES = 5
    }

    override val id = -107
    override val name = "Object Tracker (On-Device)"
    override val description = "Track and segment objects using XMem"

    private var xmemTracker: XMemTracker? = null
    private var isProcessing = AtomicBoolean(false)

    // State machine
    enum class TrackingState {
        IDLE,                  // Waiting for user trigger
        WAITING_FOR_TARGET,    // About to capture first frame
        INITIALIZING,          // Creating first-frame mask
        TRACKING,              // Active segmentation
        LOST                   // Tracking confidence too low
    }

    private var trackingState = TrackingState.IDLE
    @Volatile
    private var trackRequested = false
    @Volatile
    private var stopRequested = false

    // Tracking frame counters
    private var trackingFrameCount = 0
    private var consecutiveNullFrames = 0

    // Spatial feedback state
    private var lastSpatialFeedbackTime = 0L
    private var lastSpatialPosition = ""
    private var lastMaskArea = 0f

    // Paint objects for visualization
    private val statusPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        isFakeBoldText = true
        setShadowLayer(4f, 2f, 2f, Color.BLACK)
    }

    private val overlayPaint = Paint().apply {
        color = Color.argb(100, 0, 255, 255)  // Semi-transparent cyan
        style = Paint.Style.FILL
    }

    private val contourPaint = Paint().apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val centroidPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
    }

    private val targetPaint = Paint().apply {
        color = Color.argb(80, 255, 255, 0)  // Semi-transparent yellow
        style = Paint.Style.FILL
    }

    private val targetBorderPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    override fun initialize(context: Context) {
        try {
            xmemTracker = XMemTracker(context)
            Log.d(TAG, "VideoSegmentationProcessor initialized")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize: ${e.message}", e)
        }
    }

    override suspend fun process(frame: Bitmap): OnDeviceProcessorResult =
        withContext(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()

            // Frame dropping: skip if previous frame is still processing
            if (!isProcessing.compareAndSet(false, true)) {
                Log.d(TAG, "process() SKIPPED: isProcessing=true (stuck?)")
                return@withContext OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "",
                    processingTimeMs = 0
                )
            }

            Log.d(TAG, "process() ENTERED: state=$trackingState, trackRequested=$trackRequested")

            try {
                val tracker = xmemTracker
                if (tracker == null) {
                    isProcessing.set(false)
                    return@withContext OnDeviceProcessorResult(
                        processedImage = frame,
                        text = "Object tracker not initialized",
                        processingTimeMs = System.currentTimeMillis() - startTime
                    )
                }

                // Handle stop request
                if (stopRequested) {
                    stopRequested = false
                    tracker.reset()
                    trackingState = TrackingState.IDLE
                    lastSpatialPosition = ""
                    lastMaskArea = 0f
                    Log.d(TAG, "Tracking stopped by user")
                    isProcessing.set(false)
                    return@withContext OnDeviceProcessorResult(
                        processedImage = drawStatus(frame, "Tracking stopped. Say \"track\" to start."),
                        text = "Tracking stopped",
                        processingTimeMs = System.currentTimeMillis() - startTime
                    )
                }

                // State machine
                when (trackingState) {
                    TrackingState.IDLE -> {
                        if (trackRequested) {
                            trackRequested = false
                            trackingState = TrackingState.WAITING_FOR_TARGET
                            Log.d(TAG, "Track requested, transitioning to WAITING_FOR_TARGET")
                        }
                        isProcessing.set(false)
                        return@withContext OnDeviceProcessorResult(
                            processedImage = drawStatus(frame, "Say \"track\" to start tracking"),
                            text = "",
                            processingTimeMs = System.currentTimeMillis() - startTime
                        )
                    }

                    TrackingState.WAITING_FOR_TARGET -> {
                        // Auto-select: use center of frame as target
                        trackingState = TrackingState.INITIALIZING
                        Log.d(TAG, "Auto-selecting center of frame as target")

                        // Draw target indicator
                        val output = drawCenterTarget(frame)
                        isProcessing.set(false)
                        return@withContext OnDeviceProcessorResult(
                            processedImage = output,
                            text = "Selecting target...",
                            processingTimeMs = System.currentTimeMillis() - startTime
                        )
                    }

                    TrackingState.INITIALIZING -> {
                        // Generate first-frame mask from center region
                        val mask = generateCenterMask(frame)
                        tracker.initializeWithMask(frame, mask)
                        mask.recycle()

                        trackingState = TrackingState.TRACKING
                        trackingFrameCount = 0
                        consecutiveNullFrames = 0
                        lastSpatialFeedbackTime = System.currentTimeMillis()
                        Log.i(TAG, "Tracking initialized, entering TRACKING state")

                        isProcessing.set(false)
                        return@withContext OnDeviceProcessorResult(
                            processedImage = drawStatus(frame, "Tracking started!"),
                            text = "Tracking started",
                            processingTimeMs = System.currentTimeMillis() - startTime
                        )
                    }

                    TrackingState.TRACKING -> {
                        // Handle new track request while tracking (re-initialize)
                        if (trackRequested) {
                            trackRequested = false
                            tracker.reset()
                            trackingState = TrackingState.WAITING_FOR_TARGET
                            Log.d(TAG, "Re-track requested, resetting")
                            isProcessing.set(false)
                            return@withContext OnDeviceProcessorResult(
                                processedImage = drawStatus(frame, "Re-selecting target..."),
                                text = "",
                                processingTimeMs = System.currentTimeMillis() - startTime
                            )
                        }

                        // Run segmentation
                        val softMask = tracker.processFrame(frame)
                        trackingFrameCount++

                        // Diagnostic logging for first frames
                        if (trackingFrameCount <= GRACE_PERIOD_FRAMES) {
                            Log.d(TAG, "Tracking frame $trackingFrameCount: " +
                                "softMask=${if (softMask != null) "${softMask.size} elements" else "null"}, " +
                                "confidence=${"%.3f".format(tracker.getConfidence())}")
                        }

                        // Handle null mask (inference failed)
                        if (softMask == null) {
                            consecutiveNullFrames++
                            Log.w(TAG, "Null mask frame (consecutive=$consecutiveNullFrames/$MAX_CONSECUTIVE_NULL_FRAMES)")
                            if (consecutiveNullFrames > MAX_CONSECUTIVE_NULL_FRAMES) {
                                trackingState = TrackingState.LOST
                                Log.w(TAG, "Tracking lost: too many null frames")
                                isProcessing.set(false)
                                return@withContext OnDeviceProcessorResult(
                                    processedImage = drawStatus(frame, "Tracking lost. Say \"track\" to restart."),
                                    text = "Tracking lost. Say track to restart.",
                                    processingTimeMs = System.currentTimeMillis() - startTime
                                )
                            }
                            // Retry on next frame — show last good frame
                            isProcessing.set(false)
                            return@withContext OnDeviceProcessorResult(
                                processedImage = drawStatus(frame, "TRACKING (warming up...)"),
                                text = "",
                                processingTimeMs = System.currentTimeMillis() - startTime
                            )
                        }

                        // Valid mask received — reset null counter
                        consecutiveNullFrames = 0

                        // Check confidence only after grace period
                        if (trackingFrameCount > GRACE_PERIOD_FRAMES && tracker.isTrackingLost()) {
                            trackingState = TrackingState.LOST
                            Log.w(TAG, "Tracking lost (confidence=${tracker.getConfidence()})")
                            isProcessing.set(false)
                            return@withContext OnDeviceProcessorResult(
                                processedImage = drawStatus(frame, "Tracking lost. Say \"track\" to restart."),
                                text = "Tracking lost. Say track to restart.",
                                processingTimeMs = System.currentTimeMillis() - startTime
                            )
                        }

                        // Draw mask overlay
                        val output = drawMaskOverlay(frame, softMask, tracker.getMaskWidth(), tracker.getMaskHeight())

                        // Generate spatial feedback text
                        val spatialText = generateSpatialFeedback(
                            softMask, tracker.getMaskWidth(), tracker.getMaskHeight(),
                            frame.width, frame.height
                        )

                        isProcessing.set(false)
                        return@withContext OnDeviceProcessorResult(
                            processedImage = output,
                            text = spatialText,
                            processingTimeMs = System.currentTimeMillis() - startTime
                        )
                    }

                    TrackingState.LOST -> {
                        if (trackRequested) {
                            trackRequested = false
                            tracker.reset()
                            trackingState = TrackingState.WAITING_FOR_TARGET
                            Log.d(TAG, "Re-track from LOST state")
                        }
                        isProcessing.set(false)
                        return@withContext OnDeviceProcessorResult(
                            processedImage = drawStatus(frame, "Tracking lost. Say \"track\" to restart."),
                            text = "",
                            processingTimeMs = System.currentTimeMillis() - startTime
                        )
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Processing error: ${e.message}", e)
                isProcessing.set(false)
                return@withContext OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Error: ${e.message}",
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            }
        }

    // --- Public API for voice/button commands ---

    /**
     * Request to start tracking (called by voice "track" or photo button).
     */
    fun requestTrack() {
        trackRequested = true
        Log.d(TAG, "Track requested")
    }

    /**
     * Request to stop tracking (called by voice "stop tracking").
     */
    fun requestStopTrack() {
        stopRequested = true
        Log.d(TAG, "Stop tracking requested")
    }

    // --- First-frame mask generation ---

    /**
     * Generate a mask from the center 40% of the frame.
     * XMem will refine this rough mask within a few frames.
     */
    private fun generateCenterMask(frame: Bitmap): Bitmap {
        val mask = Bitmap.createBitmap(frame.width, frame.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(mask)
        canvas.drawColor(Color.BLACK)  // Background

        // Draw white rectangle in center 40% of frame
        val centerW = frame.width * 0.4f
        val centerH = frame.height * 0.4f
        val left = (frame.width - centerW) / 2f
        val top = (frame.height - centerH) / 2f

        val paint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.FILL
        }
        canvas.drawRect(left, top, left + centerW, top + centerH, paint)

        Log.d(TAG, "Generated center mask: ${frame.width}x${frame.height}, " +
            "rect=(${left.toInt()},${top.toInt()},${(left+centerW).toInt()},${(top+centerH).toInt()})")
        return mask
    }

    // --- Visualization ---

    /**
     * Draw segmentation mask overlay on the frame.
     */
    private fun drawMaskOverlay(
        frame: Bitmap,
        softMask: FloatArray,
        maskW: Int,
        maskH: Int
    ): Bitmap {
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val frameW = frame.width
        val frameH = frame.height

        // Scale factors from mask space to frame space
        val scaleX = frameW.toFloat() / maskW
        val scaleY = frameH.toFloat() / maskH

        // Create overlay bitmap at frame resolution
        val overlayPixels = IntArray(frameW * frameH) { Color.TRANSPARENT }

        // Apply mask with color overlay
        var centroidSumX = 0f
        var centroidSumY = 0f
        var fgCount = 0

        for (my in 0 until maskH) {
            for (mx in 0 until maskW) {
                val maskVal = softMask[my * maskW + mx]
                if (maskVal > 0.5f) {
                    // Map mask pixel to frame pixels
                    val fx = (mx * scaleX).toInt().coerceIn(0, frameW - 1)
                    val fy = (my * scaleY).toInt().coerceIn(0, frameH - 1)

                    // Fill the scaled region
                    val fx2 = ((mx + 1) * scaleX).toInt().coerceIn(0, frameW)
                    val fy2 = ((my + 1) * scaleY).toInt().coerceIn(0, frameH)

                    for (py in fy until fy2) {
                        for (px in fx until fx2) {
                            overlayPixels[py * frameW + px] = Color.argb(
                                (maskVal * 80).toInt(),  // Alpha based on confidence
                                0, 255, 255  // Cyan
                            )
                        }
                    }

                    centroidSumX += mx
                    centroidSumY += my
                    fgCount++
                }
            }
        }

        // Draw overlay
        val overlayBitmap = Bitmap.createBitmap(overlayPixels, frameW, frameH, Bitmap.Config.ARGB_8888)
        canvas.drawBitmap(overlayBitmap, 0f, 0f, null)
        overlayBitmap.recycle()

        // Draw centroid marker
        if (fgCount > 0) {
            val cx = (centroidSumX / fgCount) * scaleX
            val cy = (centroidSumY / fgCount) * scaleY
            canvas.drawCircle(cx, cy, 12f, centroidPaint)

            // Draw crosshair
            canvas.drawLine(cx - 20, cy, cx + 20, cy, contourPaint)
            canvas.drawLine(cx, cy - 20, cx, cy + 20, contourPaint)
        }

        // Draw status text
        val confidence = xmemTracker?.getConfidence() ?: 0f
        val statusText = "TRACKING (%.0f%%)".format(confidence * 100)
        canvas.drawText(statusText, 20f, 60f, statusPaint)

        return output
    }

    /**
     * Draw center target indicator (before tracking starts).
     */
    private fun drawCenterTarget(frame: Bitmap): Bitmap {
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        val centerW = frame.width * 0.4f
        val centerH = frame.height * 0.4f
        val left = (frame.width - centerW) / 2f
        val top = (frame.height - centerH) / 2f
        val rect = RectF(left, top, left + centerW, top + centerH)

        canvas.drawRect(rect, targetPaint)
        canvas.drawRect(rect, targetBorderPaint)
        canvas.drawText("Targeting center region...", 20f, 60f, statusPaint)

        return output
    }

    /**
     * Draw simple status text on frame.
     */
    private fun drawStatus(frame: Bitmap, text: String): Bitmap {
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        canvas.drawText(text, 20f, 60f, statusPaint)
        return output
    }

    // --- Spatial feedback ---

    /**
     * Generate spatial awareness TTS text based on mask position.
     * Only produces output periodically to avoid TTS spam.
     */
    private fun generateSpatialFeedback(
        softMask: FloatArray,
        maskW: Int,
        maskH: Int,
        frameW: Int,
        frameH: Int
    ): String {
        val now = System.currentTimeMillis()
        if (now - lastSpatialFeedbackTime < SPATIAL_FEEDBACK_INTERVAL_MS) {
            return ""  // Too soon for another update
        }

        // Calculate centroid and area
        var sumX = 0f
        var sumY = 0f
        var count = 0
        for (y in 0 until maskH) {
            for (x in 0 until maskW) {
                if (softMask[y * maskW + x] > 0.5f) {
                    sumX += x
                    sumY += y
                    count++
                }
            }
        }

        if (count == 0) return ""

        val centroidX = sumX / count
        val centroidY = sumY / count
        val area = count.toFloat() / (maskW * maskH)

        // Determine horizontal position
        val normalizedX = centroidX / maskW
        val position = when {
            normalizedX < 0.35f -> "left"
            normalizedX > 0.65f -> "right"
            else -> "center"
        }

        // Determine relative distance change
        val areaChange = area - lastMaskArea
        val distanceInfo = when {
            areaChange > 0.05f -> ", getting closer"
            areaChange < -0.05f -> ", getting farther"
            else -> ""
        }

        // Build feedback text
        val feedback = if (position != lastSpatialPosition || kotlin.math.abs(areaChange) > 0.05f) {
            "Object is on your $position$distanceInfo"
        } else {
            ""  // No significant change, stay silent
        }

        if (feedback.isNotEmpty()) {
            lastSpatialFeedbackTime = now
            lastSpatialPosition = position
            lastMaskArea = area
            Log.d(TAG, "Spatial feedback: $feedback (centroid=%.2f,%.2f, area=%.3f)".format(centroidX, centroidY, area))
        }

        return feedback
    }

    /**
     * Reset the processor (called by voice "stop").
     */
    fun resetTracking() {
        xmemTracker?.reset()
        trackingState = TrackingState.IDLE
        trackRequested = false
        stopRequested = false
        trackingFrameCount = 0
        consecutiveNullFrames = 0
        lastSpatialPosition = ""
        lastMaskArea = 0f
        Log.d(TAG, "Tracking reset")
    }

    override fun release() {
        xmemTracker?.release()
        xmemTracker = null
        trackingState = TrackingState.IDLE
        Log.d(TAG, "VideoSegmentationProcessor released")
    }
}
