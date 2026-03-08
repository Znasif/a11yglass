/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Feature-based homography tracker using SuperPoint + LightGlue via ONNX Runtime.
 *
 * Pipeline:
 * 1. Store reference frame when OCR runs
 * 2. Run SuperPoint+LightGlue ONNX model to match features between reference and current frame
 * 3. Compute homography matrix from matched keypoints using RANSAC
 * 4. Transform OCR bounding boxes to current frame coordinates
 */
class FeatureTracker(context: Context) {

    companion object {
        private const val TAG = "FeatureTracker"
        private const val MODEL_PATH = "models/superpoint_lightglue.onnx"
        private const val INPUT_SIZE = 512  // Model expects 512x512 images
        private const val MIN_MATCHES_FOR_HOMOGRAPHY = 4
        private const val RANSAC_THRESHOLD = 5.0f
        private const val RANSAC_ITERATIONS = 100
        // Minimum absolute inlier count required for a valid homography.
        // Prevents false positives when numPoints is small (e.g. 9 points → 33% = 3 inliers,
        // which can occur for wildly wrong shifts that happen to share 3 chance inliers).
        private const val MIN_RANSAC_INLIERS = 8
        private const val LOG_EVERY_N_FRAMES = 50
    }

    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var referenceFrame: Bitmap? = null
    private var referenceCropRect: Rect? = null
    private var isInitialized = false
    private var consecutiveFailures = 0
    private val maxConsecutiveFailures = 10

    // Current frame dimensions — set at the top of computeHomography, used by DLT normalisation
    private var currentFrameWidth = 0
    private var currentFrameHeight = 0

    // Cached preprocessed reference tensor — avoids re-preprocessing every frame
    private var cachedRefTensor: OnnxTensor? = null

    // ── Pre-allocated hot-path buffers ──────────────────────────────────────

    // Pixel scratch buffer shared between reference and current preprocessing
    private val pixelBuffer = IntArray(INPUT_SIZE * INPUT_SIZE)

    // Two float buffers: refGrayBuffer stays alive as long as cachedRefTensor is alive;
    // curGrayBuffer is overwritten each frame and only needs to survive one session.run() call.
    // Both are pre-allocated once to eliminate the ~1 MB FloatBuffer.allocate() per frame.
    private val refGrayBuffer = FloatBuffer.allocate(INPUT_SIZE * INPUT_SIZE)
    private val curGrayBuffer = FloatBuffer.allocate(INPUT_SIZE * INPUT_SIZE)

    // Match accumulator: max 512 keypoints × 4 floats (x1,y1,x2,y2)
    private val matchAccumulator = FloatArray(512 * 4)

    // RANSAC: zero-allocation random index selection and point staging
    private val ransacIndices = IntArray(4)
    private val ransacSrc     = FloatArray(8)
    private val ransacDst     = FloatArray(8)

    // DLT: pre-allocated 8×9 matrix, 8×8 Gaussian work matrix, RHS, solution, and candidate H
    // solveHomographySystem() writes into gaussX; computeHomography4Point() denormalises into candidateH.
    // The caller (RANSAC) copies candidateH only when a new best is found (~3-5× per 100 iterations).
    private val dltMatrix  = Array(8) { FloatArray(9) }
    private val gaussAA    = Array(8) { FloatArray(8) }
    private val gaussB     = FloatArray(8)
    private val gaussX     = FloatArray(8)
    private val candidateH = FloatArray(9)

    // Reusable inference input map — avoids mapOf() allocation every frame
    private val inferenceInputs = HashMap<String, OnnxTensor>(2)

    // Frame counter for periodic verbose logging
    private var frameCount = 0

    init {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            val modelBytes = context.assets.open(MODEL_PATH).use { it.readBytes() }

            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(4)
                // XNNPACK: highly optimised ARM NEON kernels for all ops in SuperPoint+LightGlue.
                // Preferred over NNAPI because:
                //   (a) NNAPI silently falls back to CPU for unsupported ops (attention layers),
                //       adding data-marshalling overhead with no acceleration benefit.
                //   (b) XNNPACK stays entirely on CPU and uses vectorised SIMD throughout.
                try {
                    addXnnpack(mapOf("intra_op_num_threads" to "4"))
                    Log.i(TAG, "XNNPACK execution provider enabled")
                } catch (e: Exception) {
                    Log.w(TAG, "XNNPACK not available, using generic CPU: ${e.message}")
                }
            }
            ortSession = ortEnvironment?.createSession(modelBytes, sessionOptions)

            Log.i(TAG, "=== ONNX MODEL METADATA ===")
            ortSession?.let { session ->
                Log.i(TAG, "Input tensors:")
                for (name in session.inputNames) {
                    Log.i(TAG, "  - '$name': ${session.inputInfo[name]?.info}")
                }
                Log.i(TAG, "Output tensors:")
                for (name in session.outputNames) {
                    Log.i(TAG, "  - '$name': ${session.outputInfo[name]?.info}")
                }
            }
            Log.i(TAG, "===========================")

            isInitialized = true
            Log.d(TAG, "FeatureTracker initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize FeatureTracker: ${e.message}", e)
        }
    }

    /**
     * Set the reference frame. Pre-computes and caches the ONNX tensor so that
     * [computeHomography] only needs to preprocess the *current* frame per call.
     *
     * [refGrayBuffer] is kept alive alongside [cachedRefTensor]; it must not be
     * reused until the tensor is closed.
     */
    fun setReferenceFrame(bitmap: Bitmap, textLabels: List<TextLabel>) {
        cachedRefTensor?.close()
        cachedRefTensor = null

        if (textLabels.isEmpty()) {
            referenceFrame = bitmap.copy(Bitmap.Config.ARGB_8888, false)
            referenceCropRect = Rect(0, 0, bitmap.width, bitmap.height)
            consecutiveFailures = 0
            Log.d(TAG, "Reference frame set (full frame): ${bitmap.width}x${bitmap.height}")
        } else {
            var unionRect = Rect(textLabels[0].boundingBox)
            for (i in 1 until textLabels.size) unionRect.union(textLabels[i].boundingBox)

            val padding = 100
            unionRect.inset(-padding, -padding)
            unionRect.intersect(0, 0, bitmap.width, bitmap.height)

            referenceCropRect = unionRect
            referenceFrame = Bitmap.createBitmap(
                bitmap, unionRect.left, unionRect.top, unionRect.width(), unionRect.height()
            )
            consecutiveFailures = 0
            Log.d(TAG, "Reference frame set (cropped): ${unionRect.width()}x${unionRect.height()} at $unionRect")
        }

        val env = ortEnvironment
        val ref = referenceFrame
        if (env != null && ref != null) {
            cachedRefTensor = preprocessImage(ref, env, refGrayBuffer)
            Log.d(TAG, "Reference tensor cached (shape=${cachedRefTensor?.info?.shape?.contentToString()})")
        }
    }

    /**
     * Compute homography between the cached reference frame and [currentFrame].
     * Returns a 3×3 row-major homography matrix, or null if tracking fails.
     */
    fun computeHomography(currentFrame: Bitmap): FloatArray? {
        if (!isInitialized || referenceFrame == null) {
            Log.w(TAG, "computeHomography: BLOCKED - isInitialized=$isInitialized, hasRefFrame=${referenceFrame != null}")
            return null
        }

        val session   = ortSession   ?: run { Log.w(TAG, "ortSession is null");   return null }
        val env       = ortEnvironment ?: run { Log.w(TAG, "ortEnvironment is null"); return null }
        val refTensor = cachedRefTensor ?: run { Log.w(TAG, "cachedRefTensor is null"); return null }

        val verbose = frameCount++ % LOG_EVERY_N_FRAMES == 0

        try {
            currentFrameWidth  = currentFrame.width
            currentFrameHeight = currentFrame.height

            var matchCount = 0

            // use{} guarantees curTensor.close() even if session.run() or parseMatchesInto() throws.
            preprocessImage(currentFrame, env, curGrayBuffer).use { curTensor ->
                if (verbose) {
                    Log.d(TAG, "computeHomography: ref=${referenceFrame!!.width}x${referenceFrame!!.height}" +
                        " cur=${currentFrame.width}x${currentFrame.height}, running inference…")
                }
                inferenceInputs["image0"] = refTensor
                inferenceInputs["image1"] = curTensor
                // use{} guarantees outputs.close() and cleanup of all child OnnxValues.
                session.run(inferenceInputs).use { outputs ->
                    matchCount = parseMatchesInto(outputs, verbose)
                }
            }

            if (matchCount < MIN_MATCHES_FOR_HOMOGRAPHY) {
                if (verbose) Log.d(TAG, "Not enough matches: $matchCount")
                consecutiveFailures++
                return null
            }

            val homography = computeHomographyRANSAC(matchAccumulator, matchCount)

            if (homography != null) {
                consecutiveFailures = 0
                if (verbose) Log.d(TAG, "Homography computed with $matchCount match pairs")
            } else {
                consecutiveFailures++
                if (verbose) Log.w(TAG, "RANSAC failed")
            }

            return homography

        } catch (e: Exception) {
            Log.e(TAG, "Error computing homography: ${e.message}", e)
            consecutiveFailures++
            return null
        }
    }

    /**
     * Transform a bounding box using the homography matrix.
     */
    fun transformBoundingBox(box: Rect, homography: FloatArray): Rect {
        val corners = floatArrayOf(
            box.left.toFloat(),  box.top.toFloat(),
            box.right.toFloat(), box.top.toFloat(),
            box.right.toFloat(), box.bottom.toFloat(),
            box.left.toFloat(),  box.bottom.toFloat()
        )

        val transformed = FloatArray(8)
        for (i in 0 until 4) {
            val x = corners[i * 2]
            val y = corners[i * 2 + 1]
            val w = homography[6] * x + homography[7] * y + homography[8]
            if (w != 0f) {
                transformed[i * 2]     = (homography[0] * x + homography[1] * y + homography[2]) / w
                transformed[i * 2 + 1] = (homography[3] * x + homography[4] * y + homography[5]) / w
            } else {
                transformed[i * 2]     = x
                transformed[i * 2 + 1] = y
            }
        }

        var minX = Float.MAX_VALUE; var minY = Float.MAX_VALUE
        var maxX = -Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (i in 0 until 4) {
            minX = minOf(minX, transformed[i * 2]);   minY = minOf(minY, transformed[i * 2 + 1])
            maxX = maxOf(maxX, transformed[i * 2]);   maxY = maxOf(maxY, transformed[i * 2 + 1])
        }
        return Rect(minX.toInt(), minY.toInt(), maxX.toInt(), maxY.toInt())
    }

    fun shouldRescan(): Boolean = consecutiveFailures >= maxConsecutiveFailures

    // ── Private implementation ───────────────────────────────────────────────

    /**
     * Preprocess [bitmap] into a grayscale [0,1] ONNX tensor using the supplied [buffer].
     *
     * The caller is responsible for managing [buffer]'s lifetime relative to the returned tensor:
     * - For the reference tensor, use [refGrayBuffer] and keep it alive as long as [cachedRefTensor].
     * - For per-frame tensors, use [curGrayBuffer]; it can be reused after the tensor is closed.
     */
    private fun preprocessImage(bitmap: Bitmap, env: OrtEnvironment, buffer: FloatBuffer): OnnxTensor {
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        scaled.getPixels(pixelBuffer, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        buffer.clear()
        for (pixel in pixelBuffer) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8)  and 0xFF
            val b =  pixel         and 0xFF
            buffer.put((0.299f * r + 0.587f * g + 0.114f * b) / 255f)
        }
        buffer.rewind()

        if (scaled != bitmap) scaled.recycle()

        return OnnxTensor.createTensor(env, buffer, longArrayOf(1, 1, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
    }

    /**
     * Parse matched keypoint pairs from ONNX output into [matchAccumulator].
     * Returns the number of valid match pairs. Zero-allocation in the hot path.
     */
    private fun parseMatchesInto(outputs: OrtSession.Result, verbose: Boolean): Int {
        try {
            val kpts0    = outputs.get("kpts0")?.get()    as? OnnxTensor
            val kpts1    = outputs.get("kpts1")?.get()    as? OnnxTensor
            val matchIdx = outputs.get("matches0")?.get() as? OnnxTensor
            val scores   = outputs.get("mscores0")?.get() as? OnnxTensor

            if (verbose) {
                Log.d(TAG, "parseMatches: kpts0=${kpts0?.info?.shape?.contentToString()}" +
                    " matches0=${matchIdx?.info?.shape?.contentToString()}" +
                    " mscores0=${scores?.info?.shape?.contentToString()}")
            }

            if (kpts0 == null || kpts1 == null || matchIdx == null) {
                Log.w(TAG, "parseMatches: expected tensors missing")
                return 0
            }

            val kpts0Buf  = kpts0.longBuffer
            val kpts1Buf  = kpts1.longBuffer
            val matchBuf  = matchIdx.longBuffer
            val scoreBuf  = scores?.floatBuffer

            val numKeypoints = matchIdx.info.shape[1].toInt()
            val refWidth     = referenceFrame!!.width.toFloat()
            val refHeight    = referenceFrame!!.height.toFloat()
            val cropOffsetX  = (referenceCropRect?.left ?: 0).toFloat()
            val cropOffsetY  = (referenceCropRect?.top  ?: 0).toFloat()
            val curWidth     = currentFrameWidth.toFloat()
            val curHeight    = currentFrameHeight.toFloat()
            val refScaleX    = refWidth  / INPUT_SIZE
            val refScaleY    = refHeight / INPUT_SIZE
            val curScaleX    = curWidth  / INPUT_SIZE
            val curScaleY    = curHeight / INPUT_SIZE

            var count = 0
            val maxMatches = matchAccumulator.size / 4

            for (i in 0 until numKeypoints) {
                val matchedIdx = matchBuf.get().toInt()
                val score      = scoreBuf?.get() ?: 1.0f
                if (matchedIdx < 0 || score < 0.5f) continue
                if (count >= maxMatches) break

                val offset = count * 4
                matchAccumulator[offset]     = kpts0Buf.get(i * 2).toFloat()         * refScaleX + cropOffsetX
                matchAccumulator[offset + 1] = kpts0Buf.get(i * 2 + 1).toFloat()     * refScaleY + cropOffsetY
                matchAccumulator[offset + 2] = kpts1Buf.get(matchedIdx * 2).toFloat() * curScaleX
                matchAccumulator[offset + 3] = kpts1Buf.get(matchedIdx * 2 + 1).toFloat() * curScaleY
                count++
            }

            if (verbose) Log.d(TAG, "parseMatches: $count valid matches out of $numKeypoints keypoints")
            return count

        } catch (e: Exception) {
            Log.e(TAG, "Error parsing matches: ${e.message}", e)
            return 0
        }
    }

    /**
     * RANSAC homography estimation.
     *
     * Zero-allocation in the per-iteration hot path:
     * - [ransacIndices] / [ransacSrc] / [ransacDst] are class-level pre-allocs.
     * - [candidateH] (filled by [computeHomography4Point]) is also pre-allocated.
     * - A heap copy (`FloatArray(9)`) is only made when a new best inlier count is found
     *   (~3-5 times across 100 iterations).
     */
    private fun computeHomographyRANSAC(matches: FloatArray, numPoints: Int): FloatArray? {
        if (numPoints < MIN_MATCHES_FOR_HOMOGRAPHY) return null

        var bestHomography: FloatArray? = null
        var bestInlierCount = 0

        repeat(RANSAC_ITERATIONS) {
            // Zero-allocation selection of 4 unique random indices via rejection sampling.
            // Expected retries ≈ 0/1/2/3 for the 4 draws — negligible overhead.
            var count = 0
            while (count < 4) {
                val r = Random.nextInt(numPoints)
                var duplicate = false
                for (j in 0 until count) { if (ransacIndices[j] == r) { duplicate = true; break } }
                if (!duplicate) ransacIndices[count++] = r
            }

            for (i in 0 until 4) {
                val idx = ransacIndices[i]
                ransacSrc[i * 2]     = matches[idx * 4]
                ransacSrc[i * 2 + 1] = matches[idx * 4 + 1]
                ransacDst[i * 2]     = matches[idx * 4 + 2]
                ransacDst[i * 2 + 1] = matches[idx * 4 + 3]
            }

            // candidateH is a class-level buffer; must copy before the next iteration overwrites it.
            if (!computeHomography4Point(ransacSrc, ransacDst)) return@repeat
            val H = candidateH

            var inlierCount = 0
            for (idx in 0 until numPoints) {
                val x1 = matches[idx * 4];     val y1 = matches[idx * 4 + 1]
                val x2 = matches[idx * 4 + 2]; val y2 = matches[idx * 4 + 3]
                val w = H[6] * x1 + H[7] * y1 + H[8]
                if (w == 0f) continue
                val px = (H[0] * x1 + H[1] * y1 + H[2]) / w
                val py = (H[3] * x1 + H[4] * y1 + H[5]) / w
                val dx = px - x2; val dy = py - y2
                if (sqrt(dx * dx + dy * dy) < RANSAC_THRESHOLD) inlierCount++
            }

            if (inlierCount > bestInlierCount) {
                bestInlierCount = inlierCount
                bestHomography = H.copyOf()  // only allocates when we genuinely improve
            }
        }

        return if (bestInlierCount >= numPoints / 3 && bestInlierCount >= MIN_RANSAC_INLIERS) bestHomography else null
    }

    /**
     * Compute homography from exactly 4 point correspondences (DLT + Gaussian elimination).
     *
     * **Normalisation:** raw pixel coordinates (up to 1280 on a typical device) produce values
     * in the millions in the `u*x` terms of the DLT matrix when mixed with the literal `±1`
     * and `0` constants. This causes catastrophic floating-point cancellation in Gaussian
     * elimination. We scale all coordinates to [0,1] using the frame's longest dimension,
     * solve in normalised space, then apply the closed-form denormalisation.
     *
     * With uniform scale `norm = 1 / max(W, H)` and `T = diag(norm, norm, 1)` for both
     * src and dst, the denormalisation is `H_orig = T⁻¹ · H_norm · T`, which reduces to:
     *   H[2] /= norm   (x-translation)
     *   H[5] /= norm   (y-translation)
     *   H[6] *= norm   (x-perspective)
     *   H[7] *= norm   (y-perspective)
     *   H[0], H[1], H[3], H[4], H[8] are unchanged (norm cancels).
     *
     * Writes the result into the pre-allocated [candidateH] buffer.
     * Returns true on success, false if the system is singular.
     */
    private fun computeHomography4Point(src: FloatArray, dst: FloatArray): Boolean {
        val norm = 1f / maxOf(currentFrameWidth, currentFrameHeight).coerceAtLeast(1)

        for (i in 0 until 4) {
            val x = src[i * 2]     * norm
            val y = src[i * 2 + 1] * norm
            val u = dst[i * 2]     * norm
            val v = dst[i * 2 + 1] * norm

            dltMatrix[i * 2].let { row ->
                row[0] = -x; row[1] = -y; row[2] = -1f
                row[3] =  0f; row[4] =  0f; row[5] =  0f
                row[6] = u * x; row[7] = u * y; row[8] = u
            }
            dltMatrix[i * 2 + 1].let { row ->
                row[0] =  0f; row[1] =  0f; row[2] =  0f
                row[3] = -x; row[4] = -y; row[5] = -1f
                row[6] = v * x; row[7] = v * y; row[8] = v
            }
        }

        if (!solveHomographySystem()) return false

        // Write solution into candidateH and append the fixed h9 = 1
        for (i in 0 until 8) candidateH[i] = gaussX[i]
        candidateH[8] = 1f

        // Denormalise (see KDoc above)
        candidateH[2] /= norm  // x-translation
        candidateH[5] /= norm  // y-translation
        candidateH[6] *= norm  // x-perspective
        candidateH[7] *= norm  // y-perspective
        // candidateH[8] = 1 and is unaffected by the uniform normalisation

        return true
    }

    /**
     * Solve the 8×8 linear system encoded in [dltMatrix] using Gaussian elimination
     * with partial pivoting. Writes the 8-element solution into [gaussX].
     *
     * Uses pre-allocated [gaussAA] and [gaussB] as work arrays — no heap allocation.
     * Row-swaps in [gaussAA] exchange FloatArray references (O(1)); the pre-allocated
     * instances are still reused on the next call because [gaussAA] is re-populated
     * from [dltMatrix] at the start of each invocation regardless of slot order.
     *
     * Returns true on success, false if the matrix is singular.
     */
    private fun solveHomographySystem(): Boolean {
        // Populate work arrays from dltMatrix (8 columns → gaussAA, 9th column → gaussB RHS)
        for (i in 0 until 8) {
            dltMatrix[i].copyInto(gaussAA[i], destinationOffset = 0, startIndex = 0, endIndex = 8)
            gaussB[i] = -dltMatrix[i][8]
        }

        // Gaussian elimination with partial pivoting
        for (col in 0 until 8) {
            var maxRow = col
            var maxVal = kotlin.math.abs(gaussAA[col][col])
            for (row in col + 1 until 8) {
                val v = kotlin.math.abs(gaussAA[row][col])
                if (v > maxVal) { maxVal = v; maxRow = row }
            }
            if (maxVal < 1e-10f) return false

            if (maxRow != col) {
                val tmp = gaussAA[col]; gaussAA[col] = gaussAA[maxRow]; gaussAA[maxRow] = tmp
                val tmpB = gaussB[col]; gaussB[col] = gaussB[maxRow]; gaussB[maxRow] = tmpB
            }

            for (row in col + 1 until 8) {
                val factor = gaussAA[row][col] / gaussAA[col][col]
                for (c in col until 8) gaussAA[row][c] -= factor * gaussAA[col][c]
                gaussB[row] -= factor * gaussB[col]
            }
        }

        // Back substitution
        for (row in 7 downTo 0) {
            var sum = gaussB[row]
            for (c in row + 1 until 8) sum -= gaussAA[row][c] * gaussX[c]
            if (kotlin.math.abs(gaussAA[row][row]) < 1e-10f) return false
            gaussX[row] = sum / gaussAA[row][row]
        }

        return true
    }

    fun release() {
        try {
            cachedRefTensor?.close()
            cachedRefTensor = null
            ortSession?.close()
            ortEnvironment?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error releasing FeatureTracker: ${e.message}")
        }
        ortSession = null
        ortEnvironment = null
        referenceFrame?.recycle()
        referenceFrame = null
        isInitialized = false
        Log.d(TAG, "FeatureTracker released")
    }
}
