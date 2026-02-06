package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.fingercount

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * On-device finger counting processor using MediaPipe Hand Landmarks.
 * Mirrors the server-side finger_count_processor.py logic.
 */
class FingerCountProcessor : OnDeviceProcessor {
    companion object {
        private const val TAG = "FingerCountProcessor"
        private const val MODEL_ASSET = "hand_landmarker.task"

        // Finger tip landmark indices
        private val TIP_IDS = intArrayOf(4, 8, 12, 16, 20)
        // Finger PIP joint indices (for comparison)
        private val PIP_IDS = intArrayOf(2, 6, 10, 14, 18)

        // Hand connection pairs for drawing
        private val HAND_CONNECTIONS = listOf(
            0 to 1, 1 to 2, 2 to 3, 3 to 4,       // Thumb
            0 to 5, 5 to 6, 6 to 7, 7 to 8,       // Index
            0 to 9, 9 to 10, 10 to 11, 11 to 12,   // Middle (via 0→9 simplification)
            0 to 13, 13 to 14, 14 to 15, 15 to 16, // Ring
            0 to 17, 17 to 18, 18 to 19, 19 to 20, // Pinky
            5 to 9, 9 to 13, 13 to 17              // Palm
        )
    }

    override val id = -101
    override val name = "Finger Count (On-Device)"
    override val description = "Count fingers using MediaPipe hand tracking"

    private var handLandmarker: HandLandmarker? = null

    private val landmarkPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        strokeWidth = 4f
    }

    private val connectionPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val textPaint = Paint().apply {
        color = Color.RED
        textSize = 120f
        isFakeBoldText = true
    }

    private val bgPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }

    override fun initialize(context: Context) {
        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .build()

            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setRunningMode(RunningMode.IMAGE)
                .build()

            handLandmarker = HandLandmarker.createFromOptions(context, options)
            Log.d(TAG, "HandLandmarker initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize HandLandmarker: ${e.message}")
        }
    }

    override suspend fun process(frame: Bitmap): OnDeviceProcessorResult =
        withContext(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()

            val landmarker = handLandmarker
            if (landmarker == null) {
                return@withContext OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Hand detector not initialized",
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            }

            try {
                val mpImage = BitmapImageBuilder(frame).build()
                val result = landmarker.detect(mpImage)

                val fingerCount = countFingers(result)
                val annotatedImage = drawResults(frame, result, fingerCount)

                val text = when {
                    result.landmarks().isNullOrEmpty() -> "No hands detected"
                    else -> "$fingerCount"
                }

                OnDeviceProcessorResult(
                    processedImage = annotatedImage,
                    text = text,
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            } catch (e: Exception) {
                Log.e(TAG, "Processing error: ${e.message}")
                OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Detection error",
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            }
        }

    /**
     * Count raised fingers across all detected hands.
     * Uses the same algorithm as the server's finger_count_processor.py.
     */
    private fun countFingers(result: HandLandmarkerResult): Int {
        if (result.landmarks().isNullOrEmpty()) return 0

        var totalFingers = 0
        for ((handIdx, handLandmarks) in result.landmarks().withIndex()) {
            val handedness = result.handedness().getOrNull(handIdx)
                ?.firstOrNull()?.categoryName() ?: "Unknown"

            // Only count Right hand (matching server behavior)
            if (handedness != "Right") continue

            for (i in TIP_IDS.indices) {
                val tip = handLandmarks[TIP_IDS[i]]
                val pip = handLandmarks[PIP_IDS[i]]

                if (i == 0) {
                    // Thumb: compare X coordinate
                    if (tip.x() > pip.x()) totalFingers++
                } else {
                    // Other fingers: tip above pip means finger is up
                    if (tip.y() < pip.y()) totalFingers++
                }
            }
        }
        return totalFingers
    }

    /**
     * Draw hand landmarks and finger count on the frame.
     */
    private fun drawResults(
        frame: Bitmap,
        result: HandLandmarkerResult,
        fingerCount: Int
    ): Bitmap {
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val width = output.width.toFloat()
        val height = output.height.toFloat()

        for (handLandmarks in result.landmarks()) {
            // Draw connections
            for ((start, end) in HAND_CONNECTIONS) {
                if (start < handLandmarks.size && end < handLandmarks.size) {
                    val startLm = handLandmarks[start]
                    val endLm = handLandmarks[end]
                    canvas.drawLine(
                        startLm.x() * width, startLm.y() * height,
                        endLm.x() * width, endLm.y() * height,
                        connectionPaint
                    )
                }
            }

            // Draw landmark points
            for (landmark in handLandmarks) {
                canvas.drawCircle(
                    landmark.x() * width,
                    landmark.y() * height,
                    6f,
                    landmarkPaint
                )
            }
        }

        // Draw finger count box (matching server behavior)
        canvas.drawRect(20f, 225f, 170f, 425f, bgPaint)
        canvas.drawText(fingerCount.toString(), 45f, 375f, textPaint)

        return output
    }

    override fun release() {
        handLandmarker?.close()
        handLandmarker = null
        Log.d(TAG, "HandLandmarker released")
    }
}
