package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.objectdetection

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
import org.tensorflow.lite.Interpreter
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * On-device object detection processor using RF-DETR via TFLite.
 * Mirrors the server-side scene_object_processor.py logic.
 */
class ObjectDetectionProcessor : OnDeviceProcessor {
    companion object {
        private const val TAG = "ObjectDetectionProcessor"
        private const val MODEL_FILE = "models/rf_detr.tflite"
        private const val LABELS_FILE = "models/coco_labels.txt"
        private const val INPUT_SIZE = 560
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val NUM_THREADS = 4
    }

    override val id = -104
    override val name = "Object Detection (On-Device)"
    override val description = "RF-DETR object detection running locally"

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var isInitialized = false

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val labelBgPaint = Paint().apply {
        style = Paint.Style.FILL
    }

    private val labelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isFakeBoldText = true
    }

    override fun initialize(context: Context) {
        try {
            // Load labels
            labels = loadLabels(context)

            // Load TFLite model
            val modelBuffer = loadModelFile(context)
            val options = Interpreter.Options().apply {
                setNumThreads(NUM_THREADS)
            }
            interpreter = Interpreter(modelBuffer, options)
            isInitialized = true
            Log.d(TAG, "RF-DETR model loaded successfully with ${labels.size} labels")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize RF-DETR: ${e.message}")
            isInitialized = false
        }
    }

    override suspend fun process(frame: Bitmap): OnDeviceProcessorResult =
        withContext(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()

            if (!isInitialized || interpreter == null) {
                return@withContext OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Object detector not initialized. Place rf_detr.tflite in assets/models/",
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            }

            try {
                // Preprocess
                val resized = Bitmap.createScaledBitmap(frame, INPUT_SIZE, INPUT_SIZE, true)
                val inputBuffer = preprocessImage(resized)
                resized.recycle()

                // Allocate output buffers
                // RF-DETR typical outputs: boxes [1, N, 4], scores [1, N], classes [1, N]
                val outputMap = allocateOutputBuffers()

                // Run inference
                interpreter!!.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

                // Post-process
                val detections = postProcess(outputMap, frame.width, frame.height)

                // Draw results
                val annotatedImage = drawDetections(frame, detections)

                // Generate text
                val text = generateText(detections)

                OnDeviceProcessorResult(
                    processedImage = annotatedImage,
                    text = text,
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            } catch (e: Exception) {
                Log.e(TAG, "Processing error: ${e.message}", e)
                OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Detection error: ${e.message}",
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            }
        }

    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            // Normalize to [0, 1]
            buffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            buffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            buffer.putFloat((pixel and 0xFF) / 255.0f)           // B
        }

        buffer.rewind()
        return buffer
    }

    private fun allocateOutputBuffers(): MutableMap<Int, Any> {
        val outputMap = mutableMapOf<Int, Any>()

        // Get output tensor info from interpreter
        val interp = interpreter!!
        val numOutputs = interp.outputTensorCount

        for (i in 0 until numOutputs) {
            val tensor = interp.getOutputTensor(i)
            val shape = tensor.shape()
            val dataType = tensor.dataType()

            val buffer = when (dataType) {
                org.tensorflow.lite.DataType.FLOAT32 -> {
                    val totalElements = shape.fold(1) { acc, dim -> acc * dim }
                    ByteBuffer.allocateDirect(totalElements * 4).order(ByteOrder.nativeOrder())
                }
                org.tensorflow.lite.DataType.INT64 -> {
                    val totalElements = shape.fold(1) { acc, dim -> acc * dim }
                    ByteBuffer.allocateDirect(totalElements * 8).order(ByteOrder.nativeOrder())
                }
                else -> {
                    val totalElements = shape.fold(1) { acc, dim -> acc * dim }
                    ByteBuffer.allocateDirect(totalElements * 4).order(ByteOrder.nativeOrder())
                }
            }
            outputMap[i] = buffer
        }

        return outputMap
    }

    private fun postProcess(
        outputMap: Map<Int, Any>,
        origWidth: Int,
        origHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()

        try {
            val interp = interpreter!!
            val numOutputs = interp.outputTensorCount
            
            // Log all output tensor info for debugging
            Log.d(TAG, "=== Model Output Debug ===")
            Log.d(TAG, "Number of outputs: $numOutputs")
            for (i in 0 until numOutputs) {
                val tensor = interp.getOutputTensor(i)
                Log.d(TAG, "Output $i: shape=${tensor.shape().contentToString()}, dtype=${tensor.dataType()}")
            }
            
            if (numOutputs < 2) {
                Log.w(TAG, "Unexpected number of outputs: $numOutputs")
                return detections
            }

            // Get output buffers
            val output0 = outputMap[0] as ByteBuffer
            val output1 = outputMap[1] as ByteBuffer
            output0.rewind()
            output1.rewind()

            val shape0 = interp.getOutputTensor(0).shape()
            val shape1 = interp.getOutputTensor(1).shape()
            val dtype0 = interp.getOutputTensor(0).dataType()
            val dtype1 = interp.getOutputTensor(1).dataType()

            Log.d(TAG, "shape0=${shape0.contentToString()}, shape1=${shape1.contentToString()}")

            // Sample first few values from each output for debugging
            val sample0 = mutableListOf<Float>()
            val sample1 = mutableListOf<Float>()
            for (i in 0 until minOf(20, output0.capacity() / 4)) {
                sample0.add(output0.getFloat())
            }
            output0.rewind()
            for (i in 0 until minOf(20, output1.capacity() / 4)) {
                sample1.add(output1.getFloat())
            }
            output1.rewind()
            Log.d(TAG, "Sample output0 (first 20): $sample0")
            Log.d(TAG, "Sample output1 (first 20): $sample1")

            // Determine which output is boxes and which is scores
            val boxesBuffer: ByteBuffer
            val scoresBuffer: ByteBuffer
            val boxesShape: IntArray
            val scoresShape: IntArray
            
            // Check if output has 4 elements in last dimension (boxes) or 1 (scores)
            if (shape0.last() == 4) {
                boxesBuffer = output0
                scoresBuffer = output1
                boxesShape = shape0
                scoresShape = shape1
                Log.d(TAG, "Output 0 is boxes, Output 1 is scores")
            } else if (shape1.last() == 4) {
                boxesBuffer = output1
                scoresBuffer = output0
                boxesShape = shape1
                scoresShape = shape0
                Log.d(TAG, "Output 1 is boxes, Output 0 is scores")
            } else {
                // Maybe the format is [batch, num_detections, 6] where 6 = [x1,y1,x2,y2,score,class]
                // or similar combined format
                Log.w(TAG, "Non-standard output format. Trying alternative parsing...")
                return parseAlternativeFormat(outputMap, origWidth, origHeight)
            }

            val numDetections = if (boxesShape.size >= 2) boxesShape[boxesShape.size - 2] else boxesShape[0]
            Log.d(TAG, "Number of potential detections: $numDetections")

            // Get class labels buffer if available
            val classLabelsBuffer = if (numOutputs >= 3) outputMap[2] as? ByteBuffer else null
            classLabelsBuffer?.rewind()

            var validCount = 0
            for (i in 0 until numDetections) {
                val score = scoresBuffer.getFloat()
                
                // Read boxes first, then check score
                val v1 = boxesBuffer.getFloat()
                val v2 = boxesBuffer.getFloat()
                val v3 = boxesBuffer.getFloat()
                val v4 = boxesBuffer.getFloat()
                
                val classLabelDtype = if (numOutputs >= 3) interp.getOutputTensor(2).dataType() else null
                val classId = classLabelsBuffer?.let {
                    when (classLabelDtype) {
                        org.tensorflow.lite.DataType.INT64 -> it.getLong().toInt()
                        org.tensorflow.lite.DataType.INT32 -> it.getInt()
                        org.tensorflow.lite.DataType.FLOAT32 -> it.getFloat().toInt()
                        else -> it.getInt() // Default to INT32
                    }
                } ?: 0

                if (score < CONFIDENCE_THRESHOLD) continue
                
                validCount++
                if (validCount <= 3) {
                    Log.d(TAG, "Det $i: score=$score, v1=$v1, v2=$v2, v3=$v3, v4=$v4, class=$classId")
                }

                // Determine if coords are normalized (0-1) or pixel coords
                // Check if values suggest cx,cy,w,h or x1,y1,x2,y2 format
                val maxVal = maxOf(v1, v2, v3, v4)
                val isNormalized = maxVal <= 1.5f
                val isCxCyWH = v3 < 0.9f && v4 < 0.9f && isNormalized // w,h usually < 0.9 if normalized
                
                val x1: Float
                val y1: Float
                val x2: Float
                val y2: Float
                
                if (isNormalized) {
                    if (isCxCyWH) {
                        // cx, cy, w, h normalized format
                        val cx = v1
                        val cy = v2
                        val w = v3
                        val h = v4
                        x1 = ((cx - w / 2) * origWidth).coerceIn(0f, origWidth.toFloat())
                        y1 = ((cy - h / 2) * origHeight).coerceIn(0f, origHeight.toFloat())
                        x2 = ((cx + w / 2) * origWidth).coerceIn(0f, origWidth.toFloat())
                        y2 = ((cy + h / 2) * origHeight).coerceIn(0f, origHeight.toFloat())
                    } else {
                        // x1, y1, x2, y2 normalized format
                        x1 = (v1 * origWidth).coerceIn(0f, origWidth.toFloat())
                        y1 = (v2 * origHeight).coerceIn(0f, origHeight.toFloat())
                        x2 = (v3 * origWidth).coerceIn(0f, origWidth.toFloat())
                        y2 = (v4 * origHeight).coerceIn(0f, origHeight.toFloat())
                    }
                } else {
                    // Already pixel coordinates, scale from input_size to orig_size
                    val scaleX = origWidth.toFloat() / INPUT_SIZE
                    val scaleY = origHeight.toFloat() / INPUT_SIZE
                    x1 = (v1 * scaleX).coerceIn(0f, origWidth.toFloat())
                    y1 = (v2 * scaleY).coerceIn(0f, origHeight.toFloat())
                    x2 = (v3 * scaleX).coerceIn(0f, origWidth.toFloat())
                    y2 = (v4 * scaleY).coerceIn(0f, origHeight.toFloat())
                }

                val className = if (classId in labels.indices) labels[classId] else "object"

                detections.add(
                    Detection(
                        bbox = RectF(x1, y1, x2, y2),
                        classId = classId,
                        className = className,
                        confidence = score
                    )
                )
            }
            
            Log.d(TAG, "Total valid detections: ${detections.size}")
        } catch (e: Exception) {
            Log.e(TAG, "Post-processing error: ${e.message}", e)
        }

        return detections
    }
    
    /**
     * Try alternative parsing for models with combined output format
     */
    private fun parseAlternativeFormat(
        outputMap: Map<Int, Any>,
        origWidth: Int,
        origHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        val interp = interpreter!!
        
        try {
            val output0 = outputMap[0] as ByteBuffer
            output0.rewind()
            val shape0 = interp.getOutputTensor(0).shape()
            
            // Common format: [1, num_detections, 6] = x1,y1,x2,y2,score,class
            // Or: [1, num_detections, 85] for YOLO with 80 classes
            if (shape0.size == 3) {
                val numDets = shape0[1]
                val elementsPerDet = shape0[2]
                
                Log.d(TAG, "Alternative format: $numDets detections x $elementsPerDet elements")
                
                for (i in 0 until numDets) {
                    if (elementsPerDet >= 6) {
                        val x1Raw = output0.getFloat()
                        val y1Raw = output0.getFloat()
                        val x2Raw = output0.getFloat()
                        val y2Raw = output0.getFloat()
                        val score = output0.getFloat()
                        val classId = output0.getFloat().toInt()
                        
                        // Skip remaining elements
                        for (j in 6 until elementsPerDet) {
                            output0.getFloat()
                        }
                        
                        if (score < CONFIDENCE_THRESHOLD) continue
                        
                        // Determine if normalized
                        val maxVal = maxOf(x1Raw, y1Raw, x2Raw, y2Raw)
                        val x1: Float
                        val y1: Float
                        val x2: Float
                        val y2: Float
                        
                        if (maxVal <= 1.5f) {
                            x1 = (x1Raw * origWidth).coerceIn(0f, origWidth.toFloat())
                            y1 = (y1Raw * origHeight).coerceIn(0f, origHeight.toFloat())
                            x2 = (x2Raw * origWidth).coerceIn(0f, origWidth.toFloat())
                            y2 = (y2Raw * origHeight).coerceIn(0f, origHeight.toFloat())
                        } else {
                            val scaleX = origWidth.toFloat() / INPUT_SIZE
                            val scaleY = origHeight.toFloat() / INPUT_SIZE
                            x1 = (x1Raw * scaleX).coerceIn(0f, origWidth.toFloat())
                            y1 = (y1Raw * scaleY).coerceIn(0f, origHeight.toFloat())
                            x2 = (x2Raw * scaleX).coerceIn(0f, origWidth.toFloat())
                            y2 = (y2Raw * scaleY).coerceIn(0f, origHeight.toFloat())
                        }
                        
                        val className = if (classId in labels.indices) labels[classId] else "object"
                        detections.add(Detection(RectF(x1, y1, x2, y2), classId, className, score))
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Alternative parsing error: ${e.message}", e)
        }
        
        return detections
    }

    private fun drawDetections(frame: Bitmap, detections: List<Detection>): Bitmap {
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        val colors = listOf(
            Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW,
            Color.CYAN, Color.MAGENTA, Color.rgb(255, 165, 0),
            Color.rgb(128, 0, 128)
        )

        for ((idx, detection) in detections.withIndex()) {
            val color = colors[idx % colors.size]
            boxPaint.color = color
            labelBgPaint.color = color

            // Draw bounding box
            canvas.drawRect(detection.bbox, boxPaint)

            // Draw label background and text
            val label = "${detection.className} ${String.format("%.2f", detection.confidence)}"
            val textWidth = labelPaint.measureText(label)
            canvas.drawRect(
                detection.bbox.left,
                detection.bbox.top - 40f,
                detection.bbox.left + textWidth + 10f,
                detection.bbox.top,
                labelBgPaint
            )
            canvas.drawText(
                label,
                detection.bbox.left + 5f,
                detection.bbox.top - 10f,
                labelPaint
            )
        }

        return output
    }

    private fun generateText(detections: List<Detection>): String {
        if (detections.isEmpty()) return "No objects detected"

        val objectCounts = detections.groupBy { it.className }
            .mapValues { it.value.size }

        return objectCounts.entries.joinToString("\n") { (label, count) ->
            if (count > 1) "$count ${label}s" else label
        }
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
        val fileInputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileInputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFileDescriptor.startOffset,
            assetFileDescriptor.declaredLength
        )
    }

    private fun loadLabels(context: Context): List<String> {
        return try {
            val reader = BufferedReader(InputStreamReader(context.assets.open(LABELS_FILE)))
            reader.readLines().filter { it.isNotBlank() }
        } catch (e: Exception) {
            Log.w(TAG, "Could not load labels file, using defaults: ${e.message}")
            defaultCocoLabels()
        }
    }

    private fun defaultCocoLabels(): List<String> = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    )

    override fun release() {
        interpreter?.close()
        interpreter = null
        isInitialized = false
        Log.d(TAG, "RF-DETR model released")
    }

    private data class Detection(
        val bbox: RectF,
        val classId: Int,
        val className: String,
        val confidence: Float
    )
}
