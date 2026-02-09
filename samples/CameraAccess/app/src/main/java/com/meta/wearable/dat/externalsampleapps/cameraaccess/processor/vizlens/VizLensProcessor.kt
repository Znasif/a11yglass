package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.tasks.await
import kotlinx.coroutines.withContext
import kotlin.math.sqrt

/**
 * VizLens Processor: Point at text to have it read aloud.
 * 
 * Pipeline:
 * 1. Scene Registration: Run OCR once to detect all text labels + bounding boxes
 * 2. Homography Tracking: Track reference image features to transform bboxes (TODO: SuperPoint+LightGlue)
 * 3. Finger Intersection: Track fingertip and check bbox intersection → TTS
 * 
 * For initial implementation, we skip homography and assume camera is relatively stable.
 * This can be enhanced with SuperPoint+LightGlue for robust tracking.
 */
class VizLensProcessor : OnDeviceProcessor {
    companion object {
        private const val TAG = "VizLensProcessor"
        private const val MODEL_ASSET = "hand_landmarker.task"
        
        // Index finger landmarks
        private const val INDEX_MCP = 5  // Base knuckle
        private const val INDEX_TIP = 8  // Fingertip
        private const val INDEX_PIP = 6  // Middle joint
        
        // Pointing detection threshold
        private const val FINGER_EXTENDED_THRESHOLD = 0.05f
    }
    
    override val id = -106
    override val name = "VizLens (On-Device)"
    override val description = "Point at text to hear it read aloud"
    
    private var handLandmarker: HandLandmarker? = null
    private var textRecognizer: TextRecognizer? = null
    private var featureTracker: FeatureTracker? = null
    private var initContext: Context? = null
    
    // State diagram: detected text labels from OCR
    private var stateDigram: List<TextLabel> = emptyList()
    private var referenceFrameRegistered = false
    
    // Track last spoken label to avoid repetition
    private var lastSpokenLabel: String? = null
    private var lastSpokenTime: Long = 0
    private val speakCooldownMs = 1500L  // Don't repeat same label for 1.5s
    
    // Store original OCR bboxes for homography transformation
    private var originalStateDiagram: List<TextLabel> = emptyList()
    private var currentHomography: FloatArray? = null
    
    // Paint for visualization
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }
    
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        isFakeBoldText = true
    }
    
    private val fingerPaint = Paint().apply {
        color = Color.CYAN
        style = Paint.Style.FILL
        strokeWidth = 6f
    }
    
    private val highlightPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    
    // Paint for crop region visualization
    private val cropRegionPaint = Paint().apply {
        color = Color.MAGENTA
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }
    
    // Crop region size (pixels) - area around fingertip to analyze
    private val cropRegionSize = 150f
    
    override fun initialize(context: Context) {
        try {
            // Initialize MediaPipe HandLandmarker
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .build()
            
            val options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setNumHands(1)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setRunningMode(RunningMode.IMAGE)
                .build()
            
            handLandmarker = HandLandmarker.createFromOptions(context, options)
            Log.d(TAG, "HandLandmarker initialized")
            
            // Initialize ML Kit Text Recognizer
            textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            Log.d(TAG, "ML Kit TextRecognizer initialized")
            
            // Initialize FeatureTracker for homography estimation
            featureTracker = FeatureTracker(context)
            initContext = context
            Log.d(TAG, "FeatureTracker initialized")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize: ${e.message}", e)
        }
    }
    
    override suspend fun process(frame: Bitmap): OnDeviceProcessorResult = 
        withContext(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()
            
            val landmarker = handLandmarker
            val recognizer = textRecognizer
            
            if (landmarker == null || recognizer == null) {
                return@withContext OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "VizLens not initialized",
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            }
            
            try {
                if (!referenceFrameRegistered || stateDigram.isEmpty()) {
                    val inputImage = InputImage.fromBitmap(frame, 0)
                    val visionText = recognizer.process(inputImage).await()
                    stateDigram = extractTextLabels(visionText)
                    originalStateDiagram = stateDigram.map { it.copy() }
                    referenceFrameRegistered = true
                    
                    // Set reference frame for homography tracking (with crop optimization)
                    featureTracker?.setReferenceFrame(frame, stateDigram)
                    
                    // Log all detected text labels for debugging
                    if (stateDigram.isNotEmpty()) {
                        Log.i(TAG, "=== OCR DETECTED TEXT ===")
                        stateDigram.forEachIndexed { index, label ->
                            Log.i(TAG, "  [$index] '${label.text}' at (${label.boundingBox.left},${label.boundingBox.top})-(${label.boundingBox.right},${label.boundingBox.bottom})")
                        }
                        Log.i(TAG, "=========================")
                    } else {
                        Log.w(TAG, "No text detected in frame")
                    }
                } else {
                    // Compute homography to track bboxes as camera moves
                    val tracker = featureTracker
                    if (tracker != null) {
                        currentHomography = tracker.computeHomography(frame)
                        
                        if (currentHomography != null) {
                            // Transform bounding boxes using homography
                            stateDigram = originalStateDiagram.map { label ->
                                TextLabel(
                                    text = label.text,
                                    boundingBox = tracker.transformBoundingBox(label.boundingBox, currentHomography!!)
                                )
                            }
                            Log.d(TAG, "Homography applied to ${stateDigram.size} bboxes")
                        } else if (tracker.shouldRescan()) {
                            // Too many tracking failures, trigger rescan
                            Log.w(TAG, "Tracking lost, triggering rescan")
                            resetScene()
                        }
                    }
                }
                
                // Phase 2: Finger Detection
                val mpImage = BitmapImageBuilder(frame).build()
                val handResult = landmarker.detect(mpImage)
                
                val fingertipPos = detectFingertip(handResult, frame.width, frame.height)
                
                // If no pointing gesture detected, just show the frame with OCR boxes (silent)
                if (fingertipPos == null) {
                    return@withContext OnDeviceProcessorResult(
                        processedImage = drawResults(frame, null, null),
                        text = "",  // Silent - no TTS output
                        processingTimeMs = System.currentTimeMillis() - startTime
                    )
                }
                
                // Phase 3: Check intersection with text bounding boxes
                val intersectedLabel = findIntersectedLabel(fingertipPos)
                
                val resultText = if (intersectedLabel != null) {
                    val now = System.currentTimeMillis()
                    if (intersectedLabel.text != lastSpokenLabel || 
                        (now - lastSpokenTime) > speakCooldownMs) {
                        lastSpokenLabel = intersectedLabel.text
                        lastSpokenTime = now
                        Log.i(TAG, ">>> SPEAKING: '${intersectedLabel.text}'")
                        intersectedLabel.text  // Return text for TTS
                    } else {
                        ""  // Don't repeat too quickly
                    }
                } else {
                    ""  // No intersection - silent
                }
                
                OnDeviceProcessorResult(
                    processedImage = drawResults(frame, fingertipPos, intersectedLabel),
                    text = resultText,
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
                
            } catch (e: Exception) {
                Log.e(TAG, "Processing error: ${e.message}", e)
                OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Error: ${e.message}",
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            }
        }
    
    /**
     * Extract text labels and bounding boxes from ML Kit OCR result.
     */
    private fun extractTextLabels(visionText: Text): List<TextLabel> {
        val labels = mutableListOf<TextLabel>()
        
        for (block in visionText.textBlocks) {
            for (line in block.lines) {
                val boundingBox = line.boundingBox
                if (boundingBox != null && line.text.isNotBlank()) {
                    labels.add(TextLabel(
                        text = line.text.trim(),
                        boundingBox = boundingBox
                    ))
                }
            }
        }
        
        return labels
    }
    
    /**
     * Detect if user is making a pointing gesture (index finger extended, others closed).
     * Uses finger ratio calculation: d/(a+b+c) where d is base-to-tip distance.
     * Ratio is 1.0 for a perfectly straight finger, lower for bent fingers.
     * 
     * Returns fingertip position in pixel coordinates, or null if not pointing.
     */
    private fun detectFingertip(
        result: HandLandmarkerResult, 
        frameWidth: Int, 
        frameHeight: Int
    ): PointF? {
        if (result.landmarks().isNullOrEmpty()) {
            Log.d(TAG, "No hand detected")
            return null
        }
        
        val handLandmarks = result.landmarks().first()
        
        // Calculate finger ratios for each finger
        // Finger landmark indices: [base, joint1, joint2, tip]
        val fingerIndices = listOf(
            listOf(1, 2, 3, 4),      // Thumb
            listOf(5, 6, 7, 8),      // Index
            listOf(9, 10, 11, 12),   // Middle
            listOf(13, 14, 15, 16),  // Ring
            listOf(17, 18, 19, 20)   // Little
        )
        
        val fingerRatios = fingerIndices.map { indices ->
            calculateFingerRatio(handLandmarks, indices)
        }
        
        // Check if pointing: index extended (>0.7), others closed (<0.95)
        val indexRatio = fingerRatios[1]
        val middleRatio = fingerRatios[2]
        val ringRatio = fingerRatios[3]
        val littleRatio = fingerRatios[4]
        
        val isPointing = indexRatio > 0.7f && 
                         middleRatio < 0.95f && 
                         ringRatio < 0.95f && 
                         littleRatio < 0.95f
        
        Log.d(TAG, "Finger ratios - Index: %.2f, Middle: %.2f, Ring: %.2f, Little: %.2f, Pointing: %s"
            .format(indexRatio, middleRatio, ringRatio, littleRatio, isPointing))
        
        if (!isPointing) {
            return null
        }
        
        // Get index fingertip position (landmark 8)
        val tip = handLandmarks[INDEX_TIP]
        return PointF(
            x = tip.x() * frameWidth,
            y = tip.y() * frameHeight
        )
    }
    
    /**
     * Calculate finger ratio: distance(base→tip) / (sum of segment distances)
     * Returns 1.0 for a perfectly straight finger, lower for bent fingers.
     */
    private fun calculateFingerRatio(
        landmarks: List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark>,
        indices: List<Int>
    ): Float {
        val p0 = landmarks[indices[0]]  // Base
        val p1 = landmarks[indices[1]]  // Joint 1
        val p2 = landmarks[indices[2]]  // Joint 2
        val p3 = landmarks[indices[3]]  // Tip
        
        // Calculate distances
        val d = distance3D(p0, p3)  // Base to tip (straight line)
        val a = distance3D(p0, p1)  // Base to joint1
        val b = distance3D(p1, p2)  // Joint1 to joint2
        val c = distance3D(p2, p3)  // Joint2 to tip
        
        val total = a + b + c
        return if (total > 0) d / total else 0f
    }
    
    /**
     * Calculate 3D Euclidean distance between two landmarks.
     */
    private fun distance3D(
        a: com.google.mediapipe.tasks.components.containers.NormalizedLandmark,
        b: com.google.mediapipe.tasks.components.containers.NormalizedLandmark
    ): Float {
        val dx = a.x() - b.x()
        val dy = a.y() - b.y()
        val dz = a.z() - b.z()
        return kotlin.math.sqrt(dx * dx + dy * dy + dz * dz)
    }
    
    /**
     * Find which text label the fingertip is pointing at.
     */
    private fun findIntersectedLabel(fingertip: PointF): TextLabel? {
        // Add some tolerance around fingertip
        val tolerance = 30f
        val fingertipRect = RectF(
            fingertip.x - tolerance,
            fingertip.y - tolerance,
            fingertip.x + tolerance,
            fingertip.y + tolerance
        )
        
        for (label in stateDigram) {
            val labelRect = RectF(label.boundingBox)
            if (RectF.intersects(fingertipRect, labelRect)) {
                return label
            }
        }
        
        return null
    }
    
    /**
     * Draw visualization: text bounding boxes, fingertip, crop region, and highlight.
     */
    private fun drawResults(
        frame: Bitmap,
        fingertip: PointF?,
        highlightedLabel: TextLabel?
    ): Bitmap {
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        
        // Draw all text bounding boxes
        for (label in stateDigram) {
            val paint = if (label == highlightedLabel) highlightPaint else boxPaint
            canvas.drawRect(label.boundingBox, paint)
        }
        
        // Draw fingertip and crop region
        fingertip?.let {
            // Draw crop region box (magenta rectangle showing scan area)
            val halfSize = cropRegionSize / 2
            canvas.drawRect(
                it.x - halfSize,
                it.y - halfSize,
                it.x + halfSize,
                it.y + halfSize,
                cropRegionPaint
            )
            
            // Draw pointing direction line (from fingertip forward)
            // For glasses view, "forward" is typically upward in the image
            val lineLength = 100f
            canvas.drawLine(it.x, it.y, it.x, it.y - lineLength, fingerPaint)
            
            // Draw fingertip dot
            canvas.drawCircle(it.x, it.y, 15f, fingerPaint)
        }
        
        // Draw status
        val statusText = when {
            stateDigram.isEmpty() -> "No text detected"
            fingertip == null -> "Show pointing finger"
            highlightedLabel != null -> "→ ${highlightedLabel.text}"
            else -> "Point at text"
        }
        canvas.drawText(statusText, 20f, 60f, textPaint)
        
        return output
    }
    
    /**
     * Force re-registration of scene (call when scene changes).
     */
    fun resetScene() {
        stateDigram = emptyList()
        originalStateDiagram = emptyList()
        referenceFrameRegistered = false
        lastSpokenLabel = null
        currentHomography = null
        Log.d(TAG, "Scene reset - will re-register on next frame")
    }
    
    override fun release() {
        handLandmarker?.close()
        handLandmarker = null
        textRecognizer?.close()
        textRecognizer = null
        featureTracker?.release()
        featureTracker = null
        stateDigram = emptyList()
        originalStateDiagram = emptyList()
        referenceFrameRegistered = false
        currentHomography = null
        Log.d(TAG, "VizLensProcessor released")
    }
}

/**
 * Represents a detected text label with its bounding box.
 */
data class TextLabel(
    val text: String,
    val boundingBox: Rect
)

/**
 * Simple 2D point for fingertip position.
 */
data class PointF(val x: Float, val y: Float)
