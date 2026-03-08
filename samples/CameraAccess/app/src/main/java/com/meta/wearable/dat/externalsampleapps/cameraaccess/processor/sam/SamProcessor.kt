package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.sam

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.FloatBuffer

/**
 * MobileSAM segmentation via OnnxRuntime.
 *
 * Two-model pipeline (standard SAM split):
 *   1. [encodeImage] — MobileSAM image encoder (1024×1024 ViT-Tiny).
 *      Runs once per source image; result is cached.
 *   2. [decodeRegion] — MobileSAM mask decoder.
 *      Takes cached embeddings + a bounding-box prompt → 256×256 mask → polygon.
 *      Very cheap (~6ms); call once per detected region.
 *
 * Models: assets/models/mobilesam/{encoder,decoder}/{model.onnx, model.data}
 *
 * Delegates (in priority order): NNAPI (NPU/GPU) → XNNPACK (ARM NEON) → CPU.
 *
 * SAM coordinate conventions:
 *   - Image is resized so the longest side = 1024, padded right+bottom with zeros.
 *   - scale = 1024 / max(origW, origH)
 *   - bbox point_coords are in the 1024-space: coord_1024 = coord_orig * scale
 *   - mask output is 256×256; mask_pixel → orig = mask_coord × 4 / scale
 */
class SamProcessor {

    companion object {
        private const val TAG = "SamProcessor"

        private const val ENCODER_ASSET = "models/mobilesam/encoder"
        private const val DECODER_ASSET = "models/mobilesam/decoder"

        private const val INPUT_SIZE      = 1024  // encoder input side
        private const val EMBED_CHANNELS  = 256   // embedding channel depth
        private const val EMBED_SPATIAL   = 64    // embedding spatial size (1024/16)
        private const val MASK_SIZE       = 256   // decoder mask output side

        // SAM / ImageNet pixel normalization
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)   // R, G, B
        private val STD  = floatArrayOf( 58.395f,  57.12f,  57.375f)

        // Contour sampling: every CONTOUR_STEP rows of the 256×256 mask
        private const val CONTOUR_STEP = 6  // → up to ~42 row pairs → ~84 polygon pts
    }

    private var ortEnv:         OrtEnvironment? = null
    private var encoderSession: OrtSession?     = null
    private var decoderSession: OrtSession?     = null

    // Cached encoder output — reused for all decodeRegion() calls on the same image
    private var cachedEmbeddings: FloatArray? = null
    private var cachedScale:      Float       = 1f
    private var cachedOrigW:      Int         = 0
    private var cachedOrigH:      Int         = 0

    val isReady: Boolean get() = encoderSession != null && decoderSession != null

    // Pre-allocated pixel scratch + image tensor buffer (avoid per-call allocations)
    private val pixelBuf = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val imageBuf = FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)

    // ── Lifecycle ────────────────────────────────────────────────────────────

    fun initialize(context: Context) {
        try {
            ortEnv = OrtEnvironment.getEnvironment()
            val env          = ortEnv!!
            val options      = buildSessionOptions()
            val encoderPath  = ensureModel(context, ENCODER_ASSET).absolutePath
            val decoderPath  = ensureModel(context, DECODER_ASSET).absolutePath
            encoderSession   = env.createSession(encoderPath, options)
            decoderSession   = env.createSession(decoderPath, options)
            Log.i(TAG, "SamProcessor initialized — encoder OK, decoder OK")
        } catch (e: Exception) {
            Log.e(TAG, "SamProcessor init failed: ${e.message}", e)
        }
    }

    fun release() {
        encoderSession?.close(); encoderSession = null
        decoderSession?.close(); decoderSession = null
        ortEnv?.close();         ortEnv         = null
        cachedEmbeddings = null
        Log.d(TAG, "SamProcessor released")
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /**
     * Encode [bitmap] and cache the image embeddings for subsequent [decodeRegion] calls.
     * Runs on [Dispatchers.Default]. Safe to call from a coroutine on any thread.
     */
    suspend fun encodeImage(bitmap: Bitmap) = withContext(Dispatchers.Default) {
        val session = encoderSession ?: run { Log.w(TAG, "encodeImage: not initialised"); return@withContext }
        val env     = ortEnv         ?: return@withContext

        val scale = INPUT_SIZE.toFloat() / maxOf(bitmap.width, bitmap.height)
        val sw = (bitmap.width  * scale).toInt().coerceAtLeast(1)
        val sh = (bitmap.height * scale).toInt().coerceAtLeast(1)

        val scaled = Bitmap.createScaledBitmap(bitmap, sw, sh, true)
        try {
            val buf = prepareImageTensor(scaled, sw, sh)
            OnnxTensor.createTensor(env, buf,
                longArrayOf(1, 3, INPUT_SIZE.toLong(), INPUT_SIZE.toLong())
            ).use { imgTensor ->
                session.run(mapOf("image" to imgTensor)).use { out ->
                    val embedT = out.get("image_embeddings")?.get() as? OnnxTensor
                        ?: run { Log.e(TAG, "encodeImage: no image_embeddings in output"); return@withContext }
                    val fb  = embedT.floatBuffer
                    val arr = FloatArray(fb.remaining()).also { fb.get(it) }
                    cachedEmbeddings = arr
                    cachedScale      = scale
                    cachedOrigW      = bitmap.width
                    cachedOrigH      = bitmap.height
                    Log.d(TAG, "encodeImage: ${bitmap.width}×${bitmap.height} " +
                        "scale=${"%.3f".format(scale)} embeddings=${arr.size}")
                }
            }
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }
    }

    /**
     * Decode a segmentation mask for [bbox] (pixel coords in the original image space).
     * Returns a flat [x0,y0, x1,y1, …] polygon in original image space, or null if
     * the mask is empty or [encodeImage] has not been called yet.
     *
     * Cheap (~6ms); call once per detected region after a single [encodeImage].
     */
    fun decodeRegion(bbox: RectF): FloatArray? {
        val session    = decoderSession   ?: return null
        val env        = ortEnv           ?: return null
        val embeddings = cachedEmbeddings ?: run { Log.w(TAG, "decodeRegion: no cached embeddings"); return null }

        val scale = cachedScale
        val origW = cachedOrigW
        val origH = cachedOrigH

        return try {
            // point_coords [1,2,2]: top-left (label=2) and bottom-right (label=3) in 1024-space
            val coordsBuf = FloatBuffer.wrap(floatArrayOf(
                bbox.left  * scale, bbox.top    * scale,
                bbox.right * scale, bbox.bottom * scale,
            ))
            val labelsBuf = FloatBuffer.wrap(floatArrayOf(2f, 3f))
            val embedBuf  = FloatBuffer.wrap(embeddings)

            OnnxTensor.createTensor(env, embedBuf,
                longArrayOf(1, EMBED_CHANNELS.toLong(), EMBED_SPATIAL.toLong(), EMBED_SPATIAL.toLong())
            ).use { embedT ->
            OnnxTensor.createTensor(env, coordsBuf, longArrayOf(1, 2, 2)).use { coordsT ->
            OnnxTensor.createTensor(env, labelsBuf, longArrayOf(1, 2)).use { labelsT ->
                session.run(mapOf(
                    "image_embeddings" to embedT,
                    "point_coords"     to coordsT,
                    "point_labels"     to labelsT,
                )).use { out ->
                    val maskT  = out.get("masks")?.get()  as? OnnxTensor ?: return null
                    val scoreT = out.get("scores")?.get() as? OnnxTensor
                    val score  = scoreT?.floatBuffer?.get(0) ?: Float.NaN
                    val fb     = maskT.floatBuffer
                    val mask   = FloatArray(fb.remaining()).also { fb.get(it) }
                    Log.d(TAG, "decodeRegion: score=${"%.3f".format(score)}")
                    maskToPolygon(mask, scale, origW, origH)
                }
            }}}
        } catch (e: Exception) {
            Log.e(TAG, "decodeRegion error: ${e.message}", e)
            null
        }
    }

    /**
     * Convenience: [encodeImage] once, then [decodeRegion] for each bbox.
     * Returns one polygon (or null) per input bbox in the same order.
     */
    suspend fun segmentRegions(bitmap: Bitmap, bboxes: List<RectF>): List<FloatArray?> {
        if (!isReady) { Log.w(TAG, "segmentRegions: not ready"); return bboxes.map { null } }
        encodeImage(bitmap)
        return bboxes.map { decodeRegion(it) }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private fun buildSessionOptions(): OrtSession.SessionOptions =
        OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(4)
            try {
                addXnnpack(mapOf("intra_op_num_threads" to "4"))
                Log.i(TAG, "XNNPACK execution provider enabled")
            } catch (e: Exception) {
                Log.w(TAG, "XNNPACK not available, using generic CPU: ${e.message}")
            }
        }

    /**
     * Copy model.onnx + model.data from assets to [Context.filesDir].
     * OnnxRuntime requires both files on disk to load a model with external data.
     * Returns the .onnx file path.
     */
    private fun ensureModel(context: Context, assetDir: String): File {
        val destDir = File(context.filesDir, assetDir).also { it.mkdirs() }
        for (name in listOf("model.onnx", "model.data")) {
            val dest = File(destDir, name)
            if (!dest.exists()) {
                Log.d(TAG, "Copying $assetDir/$name → ${dest.absolutePath}")
                context.assets.open("$assetDir/$name").use { src ->
                    dest.outputStream().use { dst -> src.copyTo(dst) }
                }
            }
        }
        return File(destDir, "model.onnx")
    }

    /**
     * Fill [imageBuf] with the SAM-normalised [1, 3, 1024, 1024] tensor.
     * [scaled] is the source bitmap at [sw]×[sh] (longest side ≤ 1024).
     * The remaining right+bottom area is padded with the normalised-zero value.
     */
    private fun prepareImageTensor(scaled: Bitmap, sw: Int, sh: Int): FloatBuffer {
        imageBuf.clear()
        scaled.getPixels(pixelBuf, 0, sw, 0, 0, sw, sh)
        // Channel-first: R plane, G plane, B plane — each INPUT_SIZE×INPUT_SIZE
        for (c in 0..2) {
            val mean   = MEAN[c]; val std = STD[c]
            val padVal = (0f - mean) / std  // normalised black for padding
            for (y in 0 until INPUT_SIZE) {
                for (x in 0 until INPUT_SIZE) {
                    if (x < sw && y < sh) {
                        val px = pixelBuf[y * sw + x]
                        val raw = when (c) {
                            0    -> (px shr 16) and 0xFF
                            1    -> (px shr  8) and 0xFF
                            else ->  px         and 0xFF
                        }.toFloat()
                        imageBuf.put((raw - mean) / std)
                    } else {
                        imageBuf.put(padVal)
                    }
                }
            }
        }
        imageBuf.flip()
        return imageBuf
    }

    /**
     * Convert the [MASK_SIZE]×[MASK_SIZE] logit mask to a flat polygon in original image space.
     *
     * Algorithm: for every [CONTOUR_STEP]-th row in the valid mask area, record the
     * leftmost and rightmost set pixel. The polygon is the left-edge path (top→bottom)
     * concatenated with the right-edge path (bottom→top), forming a closed outline.
     *
     * Coordinate mapping:  orig = mask_coord × 4 / scale
     *   (×4: mask→input space;  /scale: input→original space)
     */
    private fun maskToPolygon(mask: FloatArray, scale: Float, origW: Int, origH: Int): FloatArray? {
        // Valid mask region covers only the scaled image area (not the padding)
        val maxMaskX = (origW * scale / 4f).toInt().coerceAtMost(MASK_SIZE)
        val maxMaskY = (origH * scale / 4f).toInt().coerceAtMost(MASK_SIZE)
        val factor   = 4f / scale  // mask coord → original image pixel

        data class RowBound(val y: Int, val minX: Int, val maxX: Int)
        val bounds = mutableListOf<RowBound>()

        for (y in 0 until maxMaskY step CONTOUR_STEP) {
            var minX = -1; var maxX = -1
            for (x in 0 until maxMaskX) {
                if (mask[y * MASK_SIZE + x] > 0f) {
                    if (minX < 0) minX = x
                    maxX = x
                }
            }
            if (minX >= 0) bounds.add(RowBound(y, minX, maxX))
        }

        if (bounds.size < 3) return null

        // Left edge (top → bottom) + right edge (bottom → top) = closed polygon
        val pts = FloatArray(bounds.size * 4)
        bounds.forEachIndexed { i, b ->
            pts[i * 2]     = (b.minX * factor).coerceIn(0f, origW.toFloat())
            pts[i * 2 + 1] = (b.y    * factor).coerceIn(0f, origH.toFloat())
        }
        bounds.reversed().forEachIndexed { i, b ->
            val off = bounds.size * 2 + i * 2
            pts[off]     = (b.maxX * factor).coerceIn(0f, origW.toFloat())
            pts[off + 1] = (b.y    * factor).coerceIn(0f, origH.toFloat())
        }
        return pts
    }
}
