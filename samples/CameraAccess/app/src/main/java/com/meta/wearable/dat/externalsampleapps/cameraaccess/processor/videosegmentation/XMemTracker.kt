package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.videosegmentation

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxValue
import java.io.File
import java.nio.FloatBuffer

/**
 * XMem-based video object segmentation tracker using ONNX Runtime.
 *
 * XMem uses 4 component models:
 * 1. EncodeKeyWithShrinkage — encodes current frame keys (every frame)
 * 2. EncodeKeyWithoutShrinkage — encodes keys for first frame (once)
 * 3. EncodeValue — encodes values for memory bank (every N frames)
 * 4. Segment — segments object using memory bank (every frame)
 *
 * Memory architecture:
 * - Sensory memory: hidden state from last frame
 * - Working memory: recent key-value pairs (max 10)
 * - Long-term memory: consolidated older entries (max 50)
 */
class XMemTracker(context: Context) {

    companion object {
        private const val TAG = "XMemTracker"
        // Asset subdirectories — each contains model.onnx + model.data
        private const val ASSET_DIR_ENCODE_KEY_SHRINK = "models/xmem/encode_key_shrinkage"
        private const val ASSET_DIR_ENCODE_KEY_NO_SHRINK = "models/xmem/encode_key_no_shrinkage"
        private const val ASSET_DIR_ENCODE_VALUE = "models/xmem/encode_value"
        private const val ASSET_DIR_SEGMENT = "models/xmem/segment"
        private const val INPUT_WIDTH = 576
        private const val INPUT_HEIGHT = 320
        private const val WORKING_MEMORY_MAX_FRAMES = 10
        private const val LONG_TERM_MEMORY_MAX_FRAMES = 50
        private const val VALUE_ENCODE_INTERVAL = 5
        private const val LOG_EVERY_N_FRAMES = 50
    }

    private var ortEnvironment: OrtEnvironment? = null
    private var encodeKeyShrinkSession: OrtSession? = null
    private var encodeKeyNoShrinkSession: OrtSession? = null
    private var encodeValueSession: OrtSession? = null
    private var segmentSession: OrtSession? = null

    // Memory bank state
    private val workingMemoryKeys = mutableListOf<FloatArray>()
    private val workingMemoryValues = mutableListOf<FloatArray>()
    private val longTermMemoryKeys = mutableListOf<FloatArray>()
    private val longTermMemoryValues = mutableListOf<FloatArray>()
    private var sensoryMemory: FloatArray? = null

    // Cached shapes from first-frame inference (populated during initializeWithMask)
    private var keyShape: LongArray? = null
    private var valueShape: LongArray? = null
    private var sensoryShape: LongArray? = null

    // State
    private var frameCounter = 0
    private var isInitialized = false
    private var hasFirstFrameMask = false
    private var lastMaskLogits: FloatArray? = null

    // Reusable buffers to reduce GC pressure
    private val pixelBuffer = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
    // We need fresh FloatBuffers per tensor since ONNX Runtime takes ownership
    // but we reuse the pixel buffer for extraction

    // Tracking confidence
    private var lastConfidence = 0f

    init {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            val env = ortEnvironment!!
            Log.i(TAG, "OrtEnvironment created")

            // XMem models use ONNX external data format (model.onnx + model.data).
            // ONNX Runtime resolves .data files relative to the .onnx file path on disk,
            // so we must copy from assets to internal storage before loading.
            val modelDir = File(context.filesDir, "xmem")
            val modelSubdirs = mapOf(
                "encode_key_shrinkage" to ASSET_DIR_ENCODE_KEY_SHRINK,
                "encode_key_no_shrinkage" to ASSET_DIR_ENCODE_KEY_NO_SHRINK,
                "encode_value" to ASSET_DIR_ENCODE_VALUE,
                "segment" to ASSET_DIR_SEGMENT
            )

            for ((name, assetDir) in modelSubdirs) {
                val destDir = File(modelDir, name)
                val onnxFile = File(destDir, "model.onnx")
                if (!onnxFile.exists()) {
                    destDir.mkdirs()
                    Log.i(TAG, "Copying $name models to internal storage...")
                    copyAssetToFile(context, "$assetDir/model.onnx", onnxFile)
                    copyAssetToFile(context, "$assetDir/model.data", File(destDir, "model.data"))
                    Log.i(TAG, "  $name copied to ${destDir.absolutePath}")
                } else {
                    Log.i(TAG, "$name models already exist at ${destDir.absolutePath}")
                }
            }

            // XNNPACK-accelerated session options.
            // NNAPI must NOT be used: XMem models have 5D tensors and Android NNAPI only supports ≤4D.
            // Explicitly registering XNNPACK prevents ORT from implicitly delegating to NNAPI,
            // which can happen on some device firmware versions even without calling addNnapi().
            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(4)
                addXnnpack(emptyMap())
            }

            // Load sessions from file paths (not byte arrays) so ONNX Runtime can find .data files
            val keyShrinkPath = File(modelDir, "encode_key_shrinkage/model.onnx").absolutePath
            val keyNoShrinkPath = File(modelDir, "encode_key_no_shrinkage/model.onnx").absolutePath
            val encodeValuePath = File(modelDir, "encode_value/model.onnx").absolutePath
            val segmentPath = File(modelDir, "segment/model.onnx").absolutePath

            Log.i(TAG, "Loading EncodeKeyShrink from: $keyShrinkPath")
            encodeKeyShrinkSession = env.createSession(keyShrinkPath, sessionOptions)
            Log.i(TAG, "  EncodeKeyShrink loaded OK")

            Log.i(TAG, "Loading EncodeKeyNoShrink from: $keyNoShrinkPath")
            encodeKeyNoShrinkSession = env.createSession(keyNoShrinkPath, sessionOptions)
            Log.i(TAG, "  EncodeKeyNoShrink loaded OK")

            Log.i(TAG, "Loading EncodeValue from: $encodeValuePath")
            encodeValueSession = env.createSession(encodeValuePath, sessionOptions)
            Log.i(TAG, "  EncodeValue loaded OK")

            Log.i(TAG, "Loading Segment from: $segmentPath")
            segmentSession = env.createSession(segmentPath, sessionOptions)
            Log.i(TAG, "  Segment loaded OK")

            // Log model metadata for debugging (reveals actual input/output tensor names)
            logModelMetadata("EncodeKeyShrink", encodeKeyShrinkSession!!)
            logModelMetadata("EncodeKeyNoShrink", encodeKeyNoShrinkSession!!)
            logModelMetadata("EncodeValue", encodeValueSession!!)
            logModelMetadata("Segment", segmentSession!!)

            isInitialized = true
            Log.i(TAG, "XMemTracker initialized successfully with 4 ONNX sessions")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize XMemTracker: ${e.message}", e)
            Log.e(TAG, "Sessions loaded: keyShrink=${encodeKeyShrinkSession != null}, " +
                "keyNoShrink=${encodeKeyNoShrinkSession != null}, " +
                "encodeValue=${encodeValueSession != null}, " +
                "segment=${segmentSession != null}")
            isInitialized = false
        }
    }

    /**
     * Copy an asset file to a destination file on internal storage.
     */
    private fun copyAssetToFile(context: Context, assetPath: String, destFile: File) {
        context.assets.open(assetPath).use { input ->
            destFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }

    private fun logModelMetadata(name: String, session: OrtSession) {
        Log.i(TAG, "=== $name Model ===")
        Log.i(TAG, "  Inputs:")
        for (inputName in session.inputNames) {
            val info = session.inputInfo[inputName]
            Log.i(TAG, "    '$inputName': ${info?.info}")
        }
        Log.i(TAG, "  Outputs:")
        for (outputName in session.outputNames) {
            val info = session.outputInfo[outputName]
            Log.i(TAG, "    '$outputName': ${info?.info}")
        }
    }

    /**
     * Initialize tracking with a first-frame mask.
     * @param frame The first frame (camera resolution)
     * @param mask Binary mask (same size as frame, white=object, black=background)
     */
    fun initializeWithMask(frame: Bitmap, mask: Bitmap) {
        if (!isInitialized) {
            Log.e(TAG, "initializeWithMask: not initialized (isInitialized=false)")
            return
        }
        Log.d(TAG, "initializeWithMask: frame=${frame.width}x${frame.height}, mask=${mask.width}x${mask.height}")

        try {
            val env = ortEnvironment ?: return

            // Reset state
            clearMemoryBank()
            frameCounter = 0
            hasFirstFrameMask = false

            // Preprocess frame and mask
            val frameTensor = preprocessFrame(frame, env)
            val maskTensor = preprocessMask(mask, env)

            // Step 1: Encode key without shrinkage (first frame only)
            val keyNoShrinkSession = encodeKeyNoShrinkSession ?: return
            val keyInputs = mapOf("image" to frameTensor)
            val keyOutputs = keyNoShrinkSession.run(keyInputs)

            // Extract key tensor data
            val keyTensor = keyOutputs.getOutputTensor(0)
            val keyData = extractFloatData(keyTensor)
            keyShape = keyTensor?.info?.shape

            Log.d(TAG, "First-frame key shape: ${keyShape?.contentToString()}")

            // Step 2: Encode value for first frame
            val encodeValSession = encodeValueSession ?: return
            val valueInputs = mutableMapOf<String, OnnxTensor>()
            valueInputs["image"] = frameTensor
            valueInputs["mask"] = maskTensor

            // Add key tensor if EncodeValue expects it
            if (encodeValSession.inputNames.contains("key") && keyData != null) {
                val keyForValue = createTensorFromData(env, keyData, keyShape!!)
                valueInputs["key"] = keyForValue
            }

            val valueOutputs = encodeValSession.run(valueInputs)

            // Extract value and sensory memory
            val valueTensor = valueOutputs.getOutputTensor(0)
            val valueData = extractFloatData(valueTensor)
            valueShape = valueTensor?.info?.shape

            // Check for hidden state / sensory memory output
            if (valueOutputs.size() > 1) {
                val sensoryTensor = valueOutputs.getOutputTensor(1)
                sensoryMemory = extractFloatData(sensoryTensor)
                sensoryShape = sensoryTensor?.info?.shape
            }

            Log.d(TAG, "First-frame value shape: ${valueShape?.contentToString()}")
            Log.d(TAG, "Sensory memory shape: ${sensoryShape?.contentToString()}")

            // Initialize memory bank with first frame
            if (keyData != null) workingMemoryKeys.add(keyData)
            if (valueData != null) workingMemoryValues.add(valueData)

            // Store initial mask logits (convert mask to logits: 0→-10, 1→+10)
            lastMaskLogits = FloatArray(INPUT_WIDTH * INPUT_HEIGHT) { i ->
                val maskVal = maskTensor.floatBuffer.get(i)
                if (maskVal > 0.5f) 10f else -10f
            }

            // Cleanup
            frameTensor.close()
            maskTensor.close()
            keyOutputs.close()
            valueOutputs.close()

            hasFirstFrameMask = true
            lastConfidence = 1.0f
            Log.i(TAG, "Tracking initialized. Memory bank: ${workingMemoryKeys.size} working, ${longTermMemoryKeys.size} long-term")

        } catch (e: Exception) {
            Log.e(TAG, "Error initializing with mask: ${e.message}", e)
            hasFirstFrameMask = false
        }
    }

    /**
     * Process a new frame and return the segmentation mask.
     * @param frame Camera frame
     * @return Soft mask as FloatArray (INPUT_HEIGHT * INPUT_WIDTH), values in [0,1], or null if tracking failed
     */
    fun processFrame(frame: Bitmap): FloatArray? {
        if (!isInitialized) {
            Log.w(TAG, "processFrame: not initialized")
            return null
        }
        if (!hasFirstFrameMask) {
            Log.w(TAG, "processFrame: no first frame mask")
            return null
        }

        val verbose = frameCounter % LOG_EVERY_N_FRAMES == 0

        try {
            val env = ortEnvironment ?: return null
            val frameTensor = preprocessFrame(frame, env)

            // Step 1: Encode key with shrinkage
            val keyShrinkSession = encodeKeyShrinkSession ?: return null
            val keyInputs = mapOf("image" to frameTensor)
            val keyOutputs = keyShrinkSession.run(keyInputs)

            val keyTensor = keyOutputs.getOutputTensor(0)
            val currentKeyData = extractFloatData(keyTensor)
            val currentKeyShape = keyTensor?.info?.shape ?: keyShape

            // Extract shrinkage and selection if available
            var shrinkageData: FloatArray? = null
            var shrinkageShape: LongArray? = null
            var selectionData: FloatArray? = null
            var selectionShape: LongArray? = null

            if (keyOutputs.size() > 1) {
                val shrinkageTensor = keyOutputs.getOutputTensor(1)
                shrinkageData = extractFloatData(shrinkageTensor)
                shrinkageShape = shrinkageTensor?.info?.shape
            }
            if (keyOutputs.size() > 2) {
                val selectionTensor = keyOutputs.getOutputTensor(2)
                selectionData = extractFloatData(selectionTensor)
                selectionShape = selectionTensor?.info?.shape
            }

            // Step 2: Run Segment model
            val segSession = segmentSession ?: return null
            val segInputs = buildSegmentInputs(
                env, currentKeyData, currentKeyShape,
                shrinkageData, shrinkageShape,
                selectionData, selectionShape
            )

            val segOutputs = segSession.run(segInputs)

            // Extract mask logits
            val maskLogitsTensor = segOutputs.getOutputTensor(0)
            val maskLogits = extractFloatData(maskLogitsTensor)

            // Extract updated sensory memory (hidden state)
            if (segOutputs.size() > 1) {
                val newSensory = segOutputs.getOutputTensor(1)
                sensoryMemory = extractFloatData(newSensory)
                sensoryShape = newSensory?.info?.shape
            }

            // Apply sigmoid to get soft mask
            val softMask = if (maskLogits != null) {
                FloatArray(maskLogits.size) { i -> sigmoid(maskLogits[i]) }
            } else null

            // Calculate confidence (average mask probability for foreground pixels)
            if (softMask != null) {
                var sumFg = 0f
                var countFg = 0
                for (v in softMask) {
                    if (v > 0.5f) {
                        sumFg += v
                        countFg++
                    }
                }
                lastConfidence = if (countFg > 0) sumFg / countFg else 0f
            }

            lastMaskLogits = maskLogits

            // Step 3: Periodically encode value and update memory bank
            if (frameCounter % VALUE_ENCODE_INTERVAL == 0 && softMask != null) {
                updateMemoryBank(env, frameTensor, currentKeyData, currentKeyShape, softMask)
            }

            // Cleanup
            frameTensor.close()
            keyOutputs.close()
            // Close segment input tensors we created
            for (input in segInputs.values) {
                try { input.close() } catch (_: Exception) {}
            }
            segOutputs.close()

            frameCounter++
            if (verbose) {
                Log.d(TAG, "Frame $frameCounter: confidence=${"%.3f".format(lastConfidence)}, " +
                    "working=${workingMemoryKeys.size}, longTerm=${longTermMemoryKeys.size}")
            }

            return softMask

        } catch (e: Exception) {
            Log.e(TAG, "Error processing frame $frameCounter: ${e.message}", e)
            frameCounter++
            return null
        }
    }

    /**
     * Build the input map for the Segment model.
     * Concatenates memory bank keys/values and includes sensory memory.
     */
    private fun buildSegmentInputs(
        env: OrtEnvironment,
        currentKey: FloatArray?,
        currentKeyShape: LongArray?,
        shrinkage: FloatArray?,
        shrinkageShape: LongArray?,
        selection: FloatArray?,
        selectionShape: LongArray?
    ): Map<String, OnnxTensor> {
        val inputs = mutableMapOf<String, OnnxTensor>()
        val segSession = segmentSession ?: return inputs

        val inputNames = segSession.inputNames

        // Add current key
        if (currentKey != null && currentKeyShape != null && inputNames.contains("query_key")) {
            inputs["query_key"] = createTensorFromData(env, currentKey, currentKeyShape)
        }

        // Add shrinkage if model expects it
        if (shrinkage != null && shrinkageShape != null && inputNames.contains("query_shrinkage")) {
            inputs["query_shrinkage"] = createTensorFromData(env, shrinkage, shrinkageShape)
        }

        // Add selection if model expects it
        if (selection != null && selectionShape != null && inputNames.contains("query_selection")) {
            inputs["query_selection"] = createTensorFromData(env, selection, selectionShape)
        }

        // Concatenate memory keys and values
        val allKeys = longTermMemoryKeys + workingMemoryKeys
        val allValues = longTermMemoryValues + workingMemoryValues

        if (allKeys.isNotEmpty() && keyShape != null && inputNames.contains("memory_key")) {
            val concatenatedKeys = concatenateMemoryEntries(allKeys)
            // Memory key shape: adjust batch dim to number of memory entries
            val memKeyShape = keyShape!!.copyOf()
            if (memKeyShape.isNotEmpty()) {
                memKeyShape[0] = allKeys.size.toLong()
            }
            inputs["memory_key"] = createTensorFromData(env, concatenatedKeys, memKeyShape)
        }

        if (allValues.isNotEmpty() && valueShape != null && inputNames.contains("memory_value")) {
            val concatenatedValues = concatenateMemoryEntries(allValues)
            val memValShape = valueShape!!.copyOf()
            if (memValShape.isNotEmpty()) {
                memValShape[0] = allValues.size.toLong()
            }
            inputs["memory_value"] = createTensorFromData(env, concatenatedValues, memValShape)
        }

        // Add sensory memory (hidden state)
        if (sensoryMemory != null && sensoryShape != null && inputNames.contains("sensory_memory")) {
            inputs["sensory_memory"] = createTensorFromData(env, sensoryMemory!!, sensoryShape!!)
        }

        // Add last mask logits if model expects it
        if (lastMaskLogits != null && inputNames.contains("last_mask")) {
            val maskShape = longArrayOf(1, 1, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong())
            inputs["last_mask"] = createTensorFromData(env, lastMaskLogits!!, maskShape)
        }

        return inputs
    }

    /**
     * Update memory bank with current frame's key-value pair.
     */
    private fun updateMemoryBank(
        env: OrtEnvironment,
        frameTensor: OnnxTensor,
        keyData: FloatArray?,
        keyShape: LongArray?,
        softMask: FloatArray
    ) {
        try {
            val encodeValSession = encodeValueSession ?: return
            if (keyData == null || keyShape == null) return

            // Create mask tensor from soft mask
            val maskLogitsForEncode = FloatArray(softMask.size) { i ->
                if (softMask[i] > 0.5f) 10f else -10f
            }
            val maskShape = longArrayOf(1, 1, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong())
            val maskTensor = createTensorFromData(env, maskLogitsForEncode, maskShape)

            val valueInputs = mutableMapOf<String, OnnxTensor>()
            valueInputs["image"] = frameTensor

            // Only add mask if model expects it
            if (encodeValSession.inputNames.contains("mask")) {
                valueInputs["mask"] = maskTensor
            }
            if (encodeValSession.inputNames.contains("key")) {
                valueInputs["key"] = createTensorFromData(env, keyData, keyShape)
            }

            val valueOutputs = encodeValSession.run(valueInputs)
            val valueTensor = valueOutputs.getOutputTensor(0)
            val valueData = extractFloatData(valueTensor)

            // Update sensory memory if available
            if (valueOutputs.size() > 1) {
                val newSensory = valueOutputs.getOutputTensor(1)
                sensoryMemory = extractFloatData(newSensory)
                sensoryShape = newSensory?.info?.shape
            }

            // Add to working memory
            if (keyData != null && valueData != null) {
                workingMemoryKeys.add(keyData)
                workingMemoryValues.add(valueData)
            }

            // Consolidate if working memory is full
            if (workingMemoryKeys.size > WORKING_MEMORY_MAX_FRAMES) {
                consolidateWorkingMemory()
            }

            // Trim long-term memory if too large
            while (longTermMemoryKeys.size > LONG_TERM_MEMORY_MAX_FRAMES) {
                longTermMemoryKeys.removeAt(0)
                longTermMemoryValues.removeAt(0)
            }

            maskTensor.close()
            valueOutputs.close()

        } catch (e: Exception) {
            Log.e(TAG, "Error updating memory bank: ${e.message}", e)
        }
    }

    /**
     * Move oldest working memory entries to long-term memory.
     */
    private fun consolidateWorkingMemory() {
        // Move the oldest half of working memory to long-term
        val toMove = workingMemoryKeys.size / 2
        for (i in 0 until toMove) {
            longTermMemoryKeys.add(workingMemoryKeys.removeAt(0))
            longTermMemoryValues.add(workingMemoryValues.removeAt(0))
        }
        Log.d(TAG, "Consolidated $toMove entries. Working: ${workingMemoryKeys.size}, LongTerm: ${longTermMemoryKeys.size}")
    }

    // --- Preprocessing ---

    /**
     * Preprocess camera frame to ONNX input tensor [1, 3, H, W] normalized [0,1].
     */
    private fun preprocessFrame(bitmap: Bitmap, env: OrtEnvironment): OnnxTensor {
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_WIDTH, INPUT_HEIGHT, true)
        scaled.getPixels(pixelBuffer, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

        // CHW format: [1, 3, H, W]
        val totalElements = 3 * INPUT_HEIGHT * INPUT_WIDTH
        val floatBuffer = FloatBuffer.allocate(totalElements)

        // Channel R
        for (pixel in pixelBuffer) {
            floatBuffer.put(((pixel shr 16) and 0xFF) / 255f)
        }
        // Channel G
        for (pixel in pixelBuffer) {
            floatBuffer.put(((pixel shr 8) and 0xFF) / 255f)
        }
        // Channel B
        for (pixel in pixelBuffer) {
            floatBuffer.put((pixel and 0xFF) / 255f)
        }
        floatBuffer.rewind()

        if (scaled != bitmap) scaled.recycle()

        return OnnxTensor.createTensor(
            env, floatBuffer,
            longArrayOf(1, 3, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong())
        )
    }

    /**
     * Preprocess binary mask to ONNX tensor [1, 1, H, W] with values 0 or 1.
     */
    private fun preprocessMask(mask: Bitmap, env: OrtEnvironment): OnnxTensor {
        val scaled = Bitmap.createScaledBitmap(mask, INPUT_WIDTH, INPUT_HEIGHT, false)
        val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        scaled.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

        val floatBuffer = FloatBuffer.allocate(INPUT_WIDTH * INPUT_HEIGHT)
        for (pixel in pixels) {
            // Convert to binary: any non-black pixel is foreground
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            floatBuffer.put(if (r + g + b > 128 * 3) 1f else 0f)
        }
        floatBuffer.rewind()

        if (scaled != mask) scaled.recycle()

        return OnnxTensor.createTensor(
            env, floatBuffer,
            longArrayOf(1, 1, INPUT_HEIGHT.toLong(), INPUT_WIDTH.toLong())
        )
    }

    // --- Utility ---

    private fun extractFloatData(tensor: OnnxTensor?): FloatArray? {
        if (tensor == null) return null
        return try {
            val buf = tensor.floatBuffer
            val data = FloatArray(buf.remaining())
            buf.get(data)
            data
        } catch (e: Exception) {
            Log.w(TAG, "extractFloatData failed: ${e.message}")
            null
        }
    }

    private fun createTensorFromData(env: OrtEnvironment, data: FloatArray, shape: LongArray): OnnxTensor {
        val buffer = FloatBuffer.wrap(data)
        return OnnxTensor.createTensor(env, buffer, shape)
    }

    private fun concatenateMemoryEntries(entries: List<FloatArray>): FloatArray {
        val totalSize = entries.sumOf { it.size }
        val result = FloatArray(totalSize)
        var offset = 0
        for (entry in entries) {
            System.arraycopy(entry, 0, result, offset, entry.size)
            offset += entry.size
        }
        return result
    }

    private fun sigmoid(x: Float): Float {
        return 1f / (1f + kotlin.math.exp(-x))
    }

    /**
     * Extension to access OrtSession.Result outputs by index.
     * OrtSession.Result doesn't expose get(int) in onnxruntime-android;
     * we iterate the entries instead.
     */
    private fun OrtSession.Result.getOutputTensor(index: Int): OnnxTensor? {
        var i = 0
        for (entry in this) {
            if (i == index) return entry.value as? OnnxTensor
            i++
        }
        return null
    }

    /**
     * Get the current tracking confidence (0-1).
     * Below ~0.3 usually means tracking is lost.
     */
    fun getConfidence(): Float = lastConfidence

    /**
     * Check if tracking appears to be lost (low confidence).
     */
    fun isTrackingLost(): Boolean = lastConfidence < 0.3f

    /**
     * Get the mask dimensions.
     */
    fun getMaskWidth(): Int = INPUT_WIDTH
    fun getMaskHeight(): Int = INPUT_HEIGHT

    /**
     * Reset tracking state (clear memory bank).
     */
    fun reset() {
        clearMemoryBank()
        hasFirstFrameMask = false
        lastMaskLogits = null
        lastConfidence = 0f
        frameCounter = 0
        Log.d(TAG, "XMemTracker reset")
    }

    private fun clearMemoryBank() {
        workingMemoryKeys.clear()
        workingMemoryValues.clear()
        longTermMemoryKeys.clear()
        longTermMemoryValues.clear()
        sensoryMemory = null
    }

    /**
     * Release all resources.
     */
    fun release() {
        try {
            clearMemoryBank()
            encodeKeyShrinkSession?.close()
            encodeKeyNoShrinkSession?.close()
            encodeValueSession?.close()
            segmentSession?.close()
            ortEnvironment?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing XMemTracker: ${e.message}")
        }
        encodeKeyShrinkSession = null
        encodeKeyNoShrinkSession = null
        encodeValueSession = null
        segmentSession = null
        ortEnvironment = null
        isInitialized = false
        hasFirstFrameMask = false
        Log.d(TAG, "XMemTracker released")
    }
}
