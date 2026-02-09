/*
 * EasyOCR Engine — on-device text detection + recognition via ONNX Runtime.
 *
 * Two-stage pipeline:
 *   1. CRAFT-based text detector  → bounding boxes of text regions
 *   2. VGG-LSTM text recognizer   → character sequence per region (CTC decoded)
 *
 * Models are exported from Qualcomm AI Hub (https://aihub.qualcomm.com/models/easyocr)
 * and run on NPU via ONNX Runtime's QNN Execution Provider when available.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.FloatBuffer

class EasyOCREngine(context: Context) {

    companion object {
        private const val TAG = "EasyOCREngine"
        // Asset subdirectories — each contains model.onnx + model.data
        private const val DETECTOR_ASSET_DIR = "models/easyocr_detector"
        private const val RECOGNIZER_ASSET_DIR = "models/easyocr_recognizer"

        // CRAFT detection thresholds (from EasyOCR defaults)
        private const val TEXT_THRESHOLD = 0.7f
        private const val LINK_THRESHOLD = 0.4f
        private const val LOW_TEXT_THRESHOLD = 0.4f
        private const val MIN_COMPONENT_AREA = 10
        private const val MIN_BOX_SIZE = 20

        // ImageNet normalization for CRAFT detector
        private val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        // EasyOCR English character set.
        // Index 0 in model output = CTC blank token.
        // Index 1..N maps to CHARACTERS[0..N-1].
        private const val CHARACTERS =
            "0123456789" +
            "!\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~ " +
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    }

    // ---- ONNX state ----
    private var ortEnvironment: OrtEnvironment? = null
    private var detectorSession: OrtSession? = null
    private var recognizerSession: OrtSession? = null
    private var isInitialized = false

    // Model input dimensions — read from ONNX metadata at init time
    // so we don't hardcode values that may differ between model exports.
    private var detHeight = 0
    private var detWidth = 0
    private var recHeight = 0
    private var recWidth = 0

    init {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            val env = ortEnvironment!!

            // ONNX models with external data (model.data) must be loaded from the
            // filesystem so the runtime can resolve the relative data path.
            // Copy from assets → app internal storage on first launch.
            val detDir = copyAssetDir(context, DETECTOR_ASSET_DIR, "easyocr_detector")
            val recDir = copyAssetDir(context, RECOGNIZER_ASSET_DIR, "easyocr_recognizer")

            detectorSession = env.createSession(File(detDir, "model.onnx").absolutePath)
            recognizerSession = env.createSession(File(recDir, "model.onnx").absolutePath)

            // Read actual input dimensions from models: [batch, C, H, W]
            val detShape = readInputShape(detectorSession!!)
            detHeight = detShape[2].toInt()
            detWidth = detShape[3].toInt()

            val recShape = readInputShape(recognizerSession!!)
            recHeight = recShape[2].toInt()
            recWidth = recShape[3].toInt()

            isInitialized = true
            Log.i(TAG, "EasyOCR engine initialized — " +
                    "detector input: ${detHeight}x${detWidth}, " +
                    "recognizer input: ${recHeight}x${recWidth}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize EasyOCR engine: ${e.message}", e)
        }
    }

    /** Read the shape of a session's first input tensor. */
    private fun readInputShape(session: OrtSession): LongArray {
        val inputInfo = session.inputInfo.values.first()
        val tensorInfo = inputInfo.info as TensorInfo
        return tensorInfo.shape
    }

    /**
     * Copy an asset subdirectory (model.onnx + model.data) to app internal storage.
     * Skips files that already exist with the same size to avoid redundant copies.
     */
    private fun copyAssetDir(context: Context, assetDir: String, targetName: String): File {
        val destDir = File(context.filesDir, targetName)
        destDir.mkdirs()

        val assetFiles = context.assets.list(assetDir) ?: emptyArray()
        for (fileName in assetFiles) {
            val destFile = File(destDir, fileName)
            if (destFile.exists()) continue   // already copied
            Log.d(TAG, "Copying $assetDir/$fileName → ${destFile.absolutePath}")
            context.assets.open("$assetDir/$fileName").use { input ->
                destFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return destDir
    }

    // =========================================================================
    //  Public API
    // =========================================================================

    /**
     * Run the full OCR pipeline on [bitmap]:
     *   detect text regions → recognise each region → return text labels.
     *
     * All coordinates in the returned [TextLabel.boundingBox] are in the
     * original image coordinate space.
     */
    fun detectAndRecognize(bitmap: Bitmap): List<TextLabel> {
        if (!isInitialized) return emptyList()

        try {
            val startTime = System.currentTimeMillis()

            // 1. Detect text regions
            val boxes = runDetector(bitmap)
            if (boxes.isEmpty()) {
                Log.d(TAG, "No text regions detected")
                return emptyList()
            }
            val detMs = System.currentTimeMillis() - startTime
            Log.d(TAG, "Detected ${boxes.size} text regions in ${detMs}ms")

            // 2. Convert full image to grayscale pixel array (shared across all crops)
            val grayPixels = bitmapToGrayscaleArray(bitmap)

            // 3. Recognise text in each region
            val results = mutableListOf<TextLabel>()
            for (box in boxes) {
                val (text, confidence) = recognizeRegion(grayPixels, bitmap.width, bitmap.height, box)
                if (text.isNotBlank() && confidence > 0.1f) {
                    results.add(TextLabel(text = text, boundingBox = box))
                }
            }

            val totalMs = System.currentTimeMillis() - startTime
            Log.i(TAG, "OCR complete: ${results.size} labels in ${totalMs}ms")
            return results

        } catch (e: Exception) {
            Log.e(TAG, "OCR pipeline error: ${e.message}", e)
            return emptyList()
        }
    }

    fun release() {
        detectorSession?.close()
        recognizerSession?.close()
        // OrtEnvironment is a process-level singleton; safe to null our ref.
        detectorSession = null
        recognizerSession = null
        ortEnvironment = null
        isInitialized = false
        Log.d(TAG, "EasyOCR engine released")
    }

    // =========================================================================
    //  Detector pipeline
    // =========================================================================

    /** Metadata produced during detector preprocessing, used to map boxes back. */
    private data class DetectorMeta(
        val scale: Float,
        val padLeft: Int,
        val padTop: Int,
        val origWidth: Int,
        val origHeight: Int,
    )

    private fun runDetector(bitmap: Bitmap): List<Rect> {
        val session = detectorSession ?: return emptyList()
        val env = ortEnvironment ?: return emptyList()

        val (inputTensor, meta) = preprocessDetector(bitmap, env)

        val inputName = session.inputNames.first()
        val outputs = session.run(mapOf(inputName to inputTensor))

        val outputName = session.outputNames.first()
        val resultTensor = outputs.get(outputName)?.get() as? OnnxTensor
        if (resultTensor == null) {
            inputTensor.close(); outputs.close(); return emptyList()
        }

        val shape = resultTensor.info.shape          // [1, outH, outW, 2]
        val outH = shape[1].toInt()
        val outW = shape[2].toInt()
        val outputData = FloatArray(outH * outW * 2)
        resultTensor.floatBuffer.get(outputData)

        inputTensor.close()
        outputs.close()

        return postprocessDetector(outputData, outH, outW, meta)
    }

    /**
     * Resize + centre-pad the image to [detHeight]×[detWidth],
     * convert to NCHW float32 with ImageNet normalisation.
     */
    private fun preprocessDetector(
        bitmap: Bitmap,
        env: OrtEnvironment,
    ): Pair<OnnxTensor, DetectorMeta> {

        val scaleH = detHeight.toFloat() / bitmap.height
        val scaleW = detWidth.toFloat() / bitmap.width
        val scale = minOf(scaleH, scaleW)
        val newH = (bitmap.height * scale).toInt()
        val newW = (bitmap.width * scale).toInt()

        val padTop = (detHeight - newH) / 2
        val padLeft = (detWidth - newW) / 2
        val meta = DetectorMeta(scale, padLeft, padTop, bitmap.width, bitmap.height)

        // Resize
        val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        val pixels = IntArray(newW * newH)
        scaled.getPixels(pixels, 0, newW, 0, 0, newW, newH)
        if (scaled !== bitmap) scaled.recycle()

        // Build NCHW tensor [1, 3, 608, 800] with ImageNet normalisation.
        // Padding pixels are normalised as (0 − mean) / std.
        val padR = (-IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        val padG = (-IMAGENET_MEAN[1]) / IMAGENET_STD[1]
        val padB = (-IMAGENET_MEAN[2]) / IMAGENET_STD[2]
        val padValues = floatArrayOf(padR, padG, padB)

        val totalSize = 3 * detHeight * detWidth
        val buffer = FloatBuffer.allocate(totalSize)

        for (c in 0 until 3) {
            val planeOffset = c * detHeight * detWidth
            for (y in 0 until detHeight) {
                for (x in 0 until detWidth) {
                    val srcY = y - padTop
                    val srcX = x - padLeft
                    val value = if (srcY in 0 until newH && srcX in 0 until newW) {
                        val pixel = pixels[srcY * newW + srcX]
                        val raw = when (c) {
                            0 -> (pixel shr 16 and 0xFF) / 255f   // R
                            1 -> (pixel shr 8 and 0xFF) / 255f    // G
                            else -> (pixel and 0xFF) / 255f        // B
                        }
                        (raw - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
                    } else {
                        padValues[c]
                    }
                    buffer.put(planeOffset + y * detWidth + x, value)
                }
            }
        }
        buffer.rewind()

        val tensor = OnnxTensor.createTensor(
            env, buffer,
            longArrayOf(1, 3, detHeight.toLong(), detWidth.toLong())
        )
        return tensor to meta
    }

    /**
     * CRAFT post-processing: threshold score maps → connected components →
     * axis-aligned bounding boxes → scale back to original image coordinates.
     */
    private fun postprocessDetector(
        outputData: FloatArray,
        outH: Int,
        outW: Int,
        meta: DetectorMeta,
    ): List<Rect> {

        val n = outH * outW

        // Separate text and link score channels.
        val textScores = FloatArray(n)
        val linkScores = FloatArray(n)
        for (i in 0 until n) {
            textScores[i] = outputData[i * 2]
            linkScores[i] = outputData[i * 2 + 1]
        }

        // Binary map: combine thresholded text + link.
        val binary = BooleanArray(n)
        for (i in 0 until n) {
            val textOn = textScores[i] > LOW_TEXT_THRESHOLD
            val linkOn = linkScores[i] > LINK_THRESHOLD
            binary[i] = textOn || linkOn
        }

        // Connected-component labelling.
        val components = connectedComponentsWithStats(binary, textScores, outW, outH)

        // Scale factor: detector output is half the (padded) input resolution.
        val adjScale = meta.scale / 2f
        val adjPadLeft = meta.padLeft / 2
        val adjPadTop = meta.padTop / 2

        val boxes = mutableListOf<Rect>()
        for (comp in components) {
            if (comp.area < MIN_COMPONENT_AREA) continue
            if (comp.maxTextScore < TEXT_THRESHOLD) continue

            // Map output-space box → original-image coordinates.
            val left   = ((comp.left - adjPadLeft) / adjScale).toInt().coerceIn(0, meta.origWidth)
            val top    = ((comp.top - adjPadTop) / adjScale).toInt().coerceIn(0, meta.origHeight)
            val right  = ((comp.right + 1 - adjPadLeft) / adjScale).toInt().coerceIn(0, meta.origWidth)
            val bottom = ((comp.bottom + 1 - adjPadTop) / adjScale).toInt().coerceIn(0, meta.origHeight)

            if (maxOf(right - left, bottom - top) < MIN_BOX_SIZE) continue

            boxes.add(Rect(left, top, right, bottom))
        }

        return boxes
    }

    // =========================================================================
    //  Connected-component labelling  (two-pass, 4-connectivity, pure Kotlin)
    // =========================================================================

    private data class ComponentStats(
        var left: Int = Int.MAX_VALUE,
        var top: Int = Int.MAX_VALUE,
        var right: Int = 0,
        var bottom: Int = 0,
        var area: Int = 0,
        var maxTextScore: Float = 0f,
    )

    private fun connectedComponentsWithStats(
        binary: BooleanArray,
        textScores: FloatArray,
        width: Int,
        height: Int,
    ): List<ComponentStats> {

        val labels = IntArray(width * height) { -1 }
        val parent = mutableListOf<Int>()   // union-find
        var nextLabel = 0

        fun find(x: Int): Int {
            var r = x
            while (parent[r] != r) r = parent[r]
            var i = x
            while (i != r) { val nxt = parent[i]; parent[i] = r; i = nxt }
            return r
        }

        fun union(a: Int, b: Int) {
            val ra = find(a); val rb = find(b)
            if (ra != rb) parent[rb] = ra
        }

        // --- Pass 1: label and record equivalences ---
        for (y in 0 until height) {
            for (x in 0 until width) {
                val idx = y * width + x
                if (!binary[idx]) continue

                val leftLbl = if (x > 0 && binary[idx - 1]) labels[idx - 1] else -1
                val topLbl  = if (y > 0 && binary[idx - width]) labels[idx - width] else -1

                when {
                    leftLbl == -1 && topLbl == -1 -> {
                        labels[idx] = nextLabel
                        parent.add(nextLabel)
                        nextLabel++
                    }
                    leftLbl != -1 && topLbl == -1 -> labels[idx] = leftLbl
                    leftLbl == -1 && topLbl != -1 -> labels[idx] = topLbl
                    else -> {
                        labels[idx] = leftLbl
                        if (find(leftLbl) != find(topLbl)) union(leftLbl, topLbl)
                    }
                }
            }
        }

        // --- Pass 2: resolve labels and collect stats ---
        val statsMap = mutableMapOf<Int, ComponentStats>()
        for (y in 0 until height) {
            for (x in 0 until width) {
                val idx = y * width + x
                if (labels[idx] == -1) continue
                val root = find(labels[idx])

                val stats = statsMap.getOrPut(root) { ComponentStats() }
                if (x < stats.left)  stats.left = x
                if (y < stats.top)   stats.top = y
                if (x > stats.right) stats.right = x
                if (y > stats.bottom) stats.bottom = y
                stats.area++
                if (textScores[idx] > stats.maxTextScore)
                    stats.maxTextScore = textScores[idx]
            }
        }

        return statsMap.values.toList()
    }

    // =========================================================================
    //  Recogniser pipeline
    // =========================================================================

    /** Convert an ARGB Bitmap to a flat array of grayscale values (0..255). */
    private fun bitmapToGrayscaleArray(bitmap: Bitmap): IntArray {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        for (i in pixels.indices) {
            val p = pixels[i]
            pixels[i] = (0.299f * ((p shr 16) and 0xFF)
                       + 0.587f * ((p shr 8) and 0xFF)
                       + 0.114f * (p and 0xFF)).toInt()
        }
        return pixels
    }

    /**
     * Run the recogniser on one text region.
     * Returns (decoded text, average confidence).
     */
    private fun recognizeRegion(
        grayPixels: IntArray,
        imgWidth: Int,
        imgHeight: Int,
        box: Rect,
    ): Pair<String, Float> {
        val session = recognizerSession ?: return "" to 0f
        val env = ortEnvironment ?: return "" to 0f

        val cropW = box.width()
        val cropH = box.height()
        if (cropW <= 0 || cropH <= 0) return "" to 0f

        // Target height is recHeight; width is proportional, clamped to recWidth.
        val aspectRatio = cropW.toFloat() / cropH.toFloat()
        val newW = (recHeight * aspectRatio).toInt().coerceIn(1, recWidth)

        // Pad value: use top-left pixel of the crop as background colour.
        val tlX = box.left.coerceIn(0, imgWidth - 1)
        val tlY = box.top.coerceIn(0, imgHeight - 1)
        val padGray = grayPixels[tlY * imgWidth + tlX] / 255f
        val padNorm = (padGray - 0.5f) / 0.5f

        // Build input tensor [1, 1, 64, 800]: left-aligned, right-padded.
        val buf = FloatBuffer.allocate(recHeight * recWidth)
        // Fill entirely with pad value first.
        for (i in 0 until recHeight * recWidth) buf.put(i, padNorm)

        val scaleX = cropW.toFloat() / newW
        val scaleY = cropH.toFloat() / recHeight
        for (y in 0 until recHeight) {
            for (x in 0 until newW) {
                val srcX = (box.left + (x * scaleX).toInt()).coerceIn(0, imgWidth - 1)
                val srcY = (box.top + (y * scaleY).toInt()).coerceIn(0, imgHeight - 1)
                val gray = grayPixels[srcY * imgWidth + srcX] / 255f
                buf.put(y * recWidth + x, (gray - 0.5f) / 0.5f)
            }
        }
        buf.rewind()

        val inputTensor = OnnxTensor.createTensor(
            env, buf,
            longArrayOf(1, 1, recHeight.toLong(), recWidth.toLong())
        )

        val inputName = session.inputNames.first()
        val outputs = session.run(mapOf(inputName to inputTensor))

        val outputName = session.outputNames.first()
        val resultTensor = outputs.get(outputName)?.get() as? OnnxTensor
        if (resultTensor == null) {
            inputTensor.close(); outputs.close(); return "" to 0f
        }

        val shape = resultTensor.info.shape     // [1, T, numClasses]
        val T = shape[1].toInt()
        val C = shape[2].toInt()
        val logits = FloatArray(T * C)
        resultTensor.floatBuffer.get(logits)

        inputTensor.close()
        outputs.close()

        return ctcGreedyDecode(logits, T, C)
    }

    // =========================================================================
    //  CTC greedy decoding
    // =========================================================================

    /**
     * Greedy CTC decode: softmax → argmax → collapse repeats & remove blanks.
     * Returns (decoded string, mean confidence of non-blank predictions).
     */
    private fun ctcGreedyDecode(logits: FloatArray, T: Int, C: Int): Pair<String, Float> {
        val sb = StringBuilder()
        var confidenceSum = 0f
        var numChars = 0
        var prevIdx = 0     // 0 = blank

        for (t in 0 until T) {
            val offset = t * C

            // Numerically stable softmax: subtract max before exp.
            var maxVal = Float.NEGATIVE_INFINITY
            for (c in 0 until C) {
                val v = logits[offset + c]
                if (v > maxVal) maxVal = v
            }
            var expSum = 0f
            for (c in 0 until C) expSum += kotlin.math.exp(logits[offset + c] - maxVal)

            // Argmax + its probability.
            var bestIdx = 0
            var bestProb = 0f
            for (c in 0 until C) {
                val prob = kotlin.math.exp(logits[offset + c] - maxVal) / expSum
                if (prob > bestProb) { bestProb = prob; bestIdx = c }
            }

            // CTC rule: emit character only if non-blank and different from previous.
            if (bestIdx != 0 && bestIdx != prevIdx) {
                val charIndex = bestIdx - 1
                if (charIndex in CHARACTERS.indices) {
                    sb.append(CHARACTERS[charIndex])
                    confidenceSum += bestProb
                    numChars++
                }
            }
            prevIdx = bestIdx
        }

        val avgConf = if (numChars > 0) confidenceSum / numChars else 0f

        // Strip trailing hallucination tokens that EasyOCR can produce when
        // there is empty padding space at the right of the input.
        var text = sb.toString().trim()
        while (text.isNotEmpty() && text.last() in charArrayOf(']', '|')) {
            text = text.dropLast(1).trim()
        }

        return text to avgConf
    }
}
