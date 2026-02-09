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
    }
    
    private var ortEnvironment: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var referenceFrame: Bitmap? = null
    private var referenceCropRect: Rect? = null
    private var referenceKeypoints: FloatArray? = null
    private var isInitialized = false
    private var consecutiveFailures = 0
    private val maxConsecutiveFailures = 10
    
    // Store current frame dimensions for coordinate scaling
    private var currentFrameWidth = 0
    private var currentFrameHeight = 0
    
    init {
        try {
            ortEnvironment = OrtEnvironment.getEnvironment()
            
            val modelBytes = context.assets.open(MODEL_PATH).use { it.readBytes() }
            ortSession = ortEnvironment?.createSession(modelBytes)

            // Log model metadata for diagnostics
            Log.i(TAG, "=== ONNX MODEL METADATA ===")
            ortSession?.let { session ->
                Log.i(TAG, "Input tensors:")
                for (name in session.inputNames) {
                    val info = session.inputInfo[name]
                    Log.i(TAG, "  - '$name': ${info?.info}")
                }
                Log.i(TAG, "Output tensors:")
                for (name in session.outputNames) {
                    val info = session.outputInfo[name]
                    Log.i(TAG, "  - '$name': ${info?.info}")
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
     * Set the reference frame (called when OCR runs).
     * Extracts and uses cropped region around text for tracking.
     */
    fun setReferenceFrame(bitmap: Bitmap, textLabels: List<TextLabel>) {
        // 1. Compute union of all text bounding boxes
        if (textLabels.isEmpty()) {
            referenceFrame = bitmap.copy(Bitmap.Config.ARGB_8888, false)
            referenceCropRect = Rect(0, 0, bitmap.width, bitmap.height)
            consecutiveFailures = 0
            Log.d(TAG, "Reference frame set (full frame): ${bitmap.width}x${bitmap.height}")
            return
        }
        
        var unionRect = Rect(textLabels[0].boundingBox)
        for (i in 1 until textLabels.size) {
            unionRect.union(textLabels[i].boundingBox)
        }
        
        // 2. Add padding (e.g. 100px)
        val padding = 100
        unionRect.inset(-padding, -padding)
        
        // 3. Clip to image bounds
        unionRect.intersect(0, 0, bitmap.width, bitmap.height)
        
        // 4. Crop reference frame
        referenceCropRect = unionRect
        referenceFrame = Bitmap.createBitmap(
            bitmap, 
            unionRect.left, 
            unionRect.top, 
            unionRect.width(), 
            unionRect.height()
        )
        
        consecutiveFailures = 0
        Log.d(TAG, "Reference frame set (cropped): ${unionRect.width()}x${unionRect.height()} at $unionRect")
    }
    
    /**
     * Compute homography between reference frame and current frame.
     * Returns 3x3 homography matrix as FloatArray, or null if tracking fails.
     */
    fun computeHomography(currentFrame: Bitmap): FloatArray? {
        if (!isInitialized || referenceFrame == null) {
            Log.w(TAG, "computeHomography: BLOCKED - isInitialized=$isInitialized, hasRefFrame=${referenceFrame != null}")
            return null
        }

        val session = ortSession ?: run { Log.w(TAG, "computeHomography: ortSession is null"); return null }
        val env = ortEnvironment ?: run { Log.w(TAG, "computeHomography: ortEnvironment is null"); return null }

        try {
            currentFrameWidth = currentFrame.width
            currentFrameHeight = currentFrame.height
            Log.d(TAG, "computeHomography: refFrame=${referenceFrame!!.width}x${referenceFrame!!.height}, curFrame=${currentFrame.width}x${currentFrame.height}")

            val refTensor = preprocessImage(referenceFrame!!, env)
            val curTensor = preprocessImage(currentFrame, env)
            Log.d(TAG, "computeHomography: image0 shape=${refTensor.info.shape.contentToString()}, image1 shape=${curTensor.info.shape.contentToString()}, running inference...")

            // Run inference with two separate image inputs
            val inputs = mapOf("image0" to refTensor, "image1" to curTensor)
            val outputs = session.run(inputs)
            Log.d(TAG, "computeHomography: inference complete - ${outputs.size()} outputs returned")

            // Parse output: matched keypoint pairs
            val matches = parseMatches(outputs)
            Log.d(TAG, "computeHomography: parsed ${matches.size / 4} match pairs (${matches.size} floats)")

            // Clean up tensors
            refTensor.close()
            curTensor.close()
            outputs.close()

            if (matches.size < MIN_MATCHES_FOR_HOMOGRAPHY * 2) {
                Log.d(TAG, "Not enough matches: ${matches.size / 2}")
                consecutiveFailures++
                return null
            }

            // Compute homography using RANSAC
            val homography = computeHomographyRANSAC(matches)

            if (homography != null) {
                consecutiveFailures = 0
                Log.d(TAG, "Homography computed successfully with ${matches.size / 4} match pairs")
            } else {
                consecutiveFailures++
                Log.w(TAG, "computeHomography: RANSAC failed")
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
        // Transform all four corners
        val corners = floatArrayOf(
            box.left.toFloat(), box.top.toFloat(),
            box.right.toFloat(), box.top.toFloat(),
            box.right.toFloat(), box.bottom.toFloat(),
            box.left.toFloat(), box.bottom.toFloat()
        )
        
        val transformedCorners = FloatArray(8)
        for (i in 0 until 4) {
            val x = corners[i * 2]
            val y = corners[i * 2 + 1]
            
            // Apply homography: H * [x, y, 1]^T
            val w = homography[6] * x + homography[7] * y + homography[8]
            if (w != 0f) {
                transformedCorners[i * 2] = (homography[0] * x + homography[1] * y + homography[2]) / w
                transformedCorners[i * 2 + 1] = (homography[3] * x + homography[4] * y + homography[5]) / w
            } else {
                transformedCorners[i * 2] = x
                transformedCorners[i * 2 + 1] = y
            }
        }
        
        // Find bounding rectangle of transformed corners
        var minX = Float.MAX_VALUE
        var minY = Float.MAX_VALUE
        var maxX = Float.MIN_VALUE
        var maxY = Float.MIN_VALUE
        
        for (i in 0 until 4) {
            minX = minOf(minX, transformedCorners[i * 2])
            minY = minOf(minY, transformedCorners[i * 2 + 1])
            maxX = maxOf(maxX, transformedCorners[i * 2])
            maxY = maxOf(maxY, transformedCorners[i * 2 + 1])
        }
        
        return Rect(minX.toInt(), minY.toInt(), maxX.toInt(), maxY.toInt())
    }
    
    /**
     * Check if tracking should trigger a rescan (too many consecutive failures).
     */
    fun shouldRescan(): Boolean {
        return consecutiveFailures >= maxConsecutiveFailures
    }
    
    /**
     * Preprocess image for ONNX model input.
     * Converts to grayscale and normalizes to [0, 1].
     */
    private fun preprocessImage(bitmap: Bitmap, env: OrtEnvironment): OnnxTensor {
        // Resize to model input size
        val scaled = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        
        // Convert to grayscale float array [1, H, W]
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        
        val floatBuffer = FloatBuffer.allocate(INPUT_SIZE * INPUT_SIZE)
        for (pixel in pixels) {
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF
            val b = pixel and 0xFF
            // Grayscale conversion: 0.299*R + 0.587*G + 0.114*B, normalized to [0,1]
            val gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255f
            floatBuffer.put(gray)
        }
        floatBuffer.rewind()
        
        if (scaled != bitmap) {
            scaled.recycle()
        }
        
        return OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 1, INPUT_SIZE.toLong(), INPUT_SIZE.toLong()))
    }
    
    /**
     * Parse matched keypoint pairs from ONNX output.
     * Returns flattened array: [x1, y1, x2, y2, ...] where (x1,y1) is reference point, (x2,y2) is current point.
     */
    private fun parseMatches(outputs: OrtSession.Result): FloatArray {
        // Actual model output format:
        //   kpts0:    INT64 [1, N, 2]  — N keypoint (x,y) coords in image0 (model input space)
        //   kpts1:    INT64 [1, M, 2]  — M keypoint (x,y) coords in image1 (model input space)
        //   matches0: INT64 [1, N]     — for each kpt in image0, index of matching kpt in image1 (-1 = unmatched)
        //   mscores0: FLOAT [1, N]     — confidence score for each match

        try {
            // Log all available output names
            val outputNames = outputs.map { it.key }
            Log.d(TAG, "parseMatches: available outputs: $outputNames")

            val kpts0 = outputs.get("kpts0")?.get() as? OnnxTensor
            val kpts1 = outputs.get("kpts1")?.get() as? OnnxTensor
            val matchIdx = outputs.get("matches0")?.get() as? OnnxTensor
            val scores = outputs.get("mscores0")?.get() as? OnnxTensor

            Log.d(TAG, "parseMatches: kpts0=${kpts0?.info?.shape?.contentToString() ?: "NULL"} (${kpts0?.info?.type}), kpts1=${kpts1?.info?.shape?.contentToString() ?: "NULL"} (${kpts1?.info?.type}), matches0=${matchIdx?.info?.shape?.contentToString() ?: "NULL"}, mscores0=${scores?.info?.shape?.contentToString() ?: "NULL"}")

            if (kpts0 == null || kpts1 == null || matchIdx == null) {
                Log.w(TAG, "parseMatches: expected tensors missing, falling back to alternate parsing")
                return parseMatchesAlternate(outputs)
            }

            // Keypoints are INT64 pixel coordinates in model input space (0..INPUT_SIZE-1)
            val kpts0Buf = kpts0.longBuffer    // [1, N, 2] flattened: [x0,y0, x1,y1, ...] for N keypoints
            val kpts1Buf = kpts1.longBuffer    // [1, M, 2] flattened: [x0,y0, x1,y1, ...] for M keypoints
            val matchBuf = matchIdx.longBuffer // [1, N] flattened: for each kpt i in image0, matchBuf[i] = matched kpt index in image1 (-1 = no match)
            val scoreBuf = scores?.floatBuffer // [1, N] flattened: confidence for each kpt's match

            // Number of keypoints in image0 (the dimension we iterate over)
            val numKeypoints = matchIdx.info.shape[1].toInt()
            val refWidth = referenceFrame!!.width.toFloat()
            val refHeight = referenceFrame!!.height.toFloat()
            val cropOffsetX = (referenceCropRect?.left ?: 0).toFloat()
            val cropOffsetY = (referenceCropRect?.top ?: 0).toFloat()
            val curWidth = currentFrameWidth.toFloat()
            val curHeight = currentFrameHeight.toFloat()

            Log.d(TAG, "parseMatches: $numKeypoints keypoints in image0, refSize=${refWidth}x${refHeight}, cropOffset=($cropOffsetX,$cropOffsetY), curSize=${curWidth}x${curHeight}")

            val matches = mutableListOf<Float>()

            for (i in 0 until numKeypoints) {
                val matchedIdx = matchBuf.get().toInt()  // index into kpts1, or -1 if unmatched
                val score = scoreBuf?.get() ?: 1.0f
                if (matchedIdx < 0 || score < 0.2f) continue

                // kpts0[i] coords: pixel coords in model input space → scale to reference crop, add crop offset
                val x0_model = kpts0Buf.get(i * 2).toFloat()
                val y0_model = kpts0Buf.get(i * 2 + 1).toFloat()
                val x0_global = x0_model * refWidth / INPUT_SIZE + cropOffsetX
                val y0_global = y0_model * refHeight / INPUT_SIZE + cropOffsetY

                // kpts1[matchedIdx] coords: pixel coords in model input space → scale to full current frame
                val x1_model = kpts1Buf.get(matchedIdx * 2).toFloat()
                val y1_model = kpts1Buf.get(matchedIdx * 2 + 1).toFloat()
                val x1_global = x1_model * curWidth / INPUT_SIZE
                val y1_global = y1_model * curHeight / INPUT_SIZE

                matches.add(x0_global)
                matches.add(y0_global)
                matches.add(x1_global)
                matches.add(y1_global)
            }

            Log.d(TAG, "parseMatches: ${matches.size / 4} valid matches out of $numKeypoints keypoints")
            return matches.toFloatArray()

        } catch (e: Exception) {
            Log.e(TAG, "Error parsing matches: ${e.message}", e)
            return floatArrayOf()
        }
    }

    /**
     * Alternate parsing for different model output formats.
     * Dumps comprehensive output information for diagnostics.
     */
    private fun parseMatchesAlternate(outputs: OrtSession.Result): FloatArray {
        Log.w(TAG, "parseMatchesAlternate: PRIMARY PARSING FAILED - dumping all outputs")
        for ((name, value) in outputs) {
            val tensor = value as? OnnxTensor
            if (tensor != null) {
                val shape = tensor.info.shape
                Log.w(TAG, "  Output '$name': shape=${shape.contentToString()}, type=${tensor.info.type}")
                try {
                    val buf = tensor.floatBuffer
                    val preview = FloatArray(minOf(10, buf.remaining()))
                    for (i in preview.indices) preview[i] = buf.get()
                    buf.rewind()
                    Log.w(TAG, "  Preview (float): ${preview.contentToString()}")
                } catch (_: Exception) {
                    try {
                        val buf = tensor.longBuffer
                        val preview = LongArray(minOf(10, buf.remaining()))
                        for (i in preview.indices) preview[i] = buf.get()
                        buf.rewind()
                        Log.w(TAG, "  Preview (long): ${preview.contentToString()}")
                    } catch (_: Exception) {
                        Log.w(TAG, "  Could not preview values")
                    }
                }
            } else {
                Log.w(TAG, "  Output '$name': not OnnxTensor (${value?.javaClass?.simpleName})")
            }
        }
        Log.e(TAG, "parseMatchesAlternate: RETURNING EMPTY - model output format incompatible")
        return floatArrayOf()
    }
    
    /**
     * Compute homography using RANSAC algorithm.
     * Input: matches array [x1, y1, x2, y2, ...] where (x1,y1) is reference, (x2,y2) is current.
     */
    private fun computeHomographyRANSAC(matches: FloatArray): FloatArray? {
        val numPoints = matches.size / 4
        Log.d(TAG, "RANSAC: $numPoints point correspondences, need >= $MIN_MATCHES_FOR_HOMOGRAPHY")
        if (numPoints < MIN_MATCHES_FOR_HOMOGRAPHY) {
            Log.w(TAG, "RANSAC: INSUFFICIENT POINTS")
            return null
        }
        
        var bestHomography: FloatArray? = null
        var bestInlierCount = 0
        
        repeat(RANSAC_ITERATIONS) {
            // Sample 4 random points
            val indices = (0 until numPoints).shuffled().take(4)
            
            // Extract point correspondences
            val srcPoints = FloatArray(8)
            val dstPoints = FloatArray(8)
            
            for ((i, idx) in indices.withIndex()) {
                srcPoints[i * 2] = matches[idx * 4]         // x1
                srcPoints[i * 2 + 1] = matches[idx * 4 + 1] // y1
                dstPoints[i * 2] = matches[idx * 4 + 2]     // x2
                dstPoints[i * 2 + 1] = matches[idx * 4 + 3] // y2
            }
            
            // Compute homography from 4-point correspondences
            val H = computeHomography4Point(srcPoints, dstPoints) ?: return@repeat
            
            // Count inliers
            var inlierCount = 0
            for (idx in 0 until numPoints) {
                val x1 = matches[idx * 4]
                val y1 = matches[idx * 4 + 1]
                val x2 = matches[idx * 4 + 2]
                val y2 = matches[idx * 4 + 3]
                
                // Project x1,y1 using H
                val w = H[6] * x1 + H[7] * y1 + H[8]
                if (w == 0f) continue
                val px = (H[0] * x1 + H[1] * y1 + H[2]) / w
                val py = (H[3] * x1 + H[4] * y1 + H[5]) / w
                
                // Check reprojection error
                val dx = px - x2
                val dy = py - y2
                val error = sqrt(dx * dx + dy * dy)
                
                if (error < RANSAC_THRESHOLD) {
                    inlierCount++
                }
            }
            
            if (inlierCount > bestInlierCount) {
                bestInlierCount = inlierCount
                bestHomography = H
            }
        }
        
        Log.d(TAG, "RANSAC: best inliers=$bestInlierCount/$numPoints (${if (numPoints > 0) bestInlierCount * 100 / numPoints else 0}%), need 50%")

        // Need at least 50% inliers
        if (bestInlierCount < numPoints / 2) {
            Log.w(TAG, "RANSAC: FAILED - not enough inliers")
            return null
        }

        Log.i(TAG, "RANSAC: SUCCESS")
        return bestHomography
    }
    
    /**
     * Compute homography from exactly 4 point correspondences using Direct Linear Transform (DLT).
     */
    private fun computeHomography4Point(src: FloatArray, dst: FloatArray): FloatArray? {
        // Build the 8x9 matrix A for Ah = 0
        val A = Array(8) { FloatArray(9) }
        
        for (i in 0 until 4) {
            val x = src[i * 2]
            val y = src[i * 2 + 1]
            val u = dst[i * 2]
            val v = dst[i * 2 + 1]
            
            A[i * 2] = floatArrayOf(-x, -y, -1f, 0f, 0f, 0f, u * x, u * y, u)
            A[i * 2 + 1] = floatArrayOf(0f, 0f, 0f, -x, -y, -1f, v * x, v * y, v)
        }
        
        // Solve using simplified approach (assume last element is 1)
        // This is a simplified version - full DLT would use SVD
        val H = solveHomographySystem(A) ?: return null
        
        // Normalize so H[8] = 1
        if (H[8] != 0f) {
            for (i in 0 until 9) {
                H[i] /= H[8]
            }
        }
        
        return H
    }
    
    /**
     * Solve the homography linear system (simplified Gaussian elimination).
     */
    private fun solveHomographySystem(A: Array<FloatArray>): FloatArray? {
        // Use a simplified approach: set h9 = 1 and solve 8x8 system
        // For production, SVD would be more robust
        
        // Create 8x8 system by moving the 9th column to RHS
        val AA = Array(8) { i -> A[i].sliceArray(0 until 8) }
        val b = FloatArray(8) { i -> -A[i][8] }
        
        // Gaussian elimination with partial pivoting
        for (col in 0 until 8) {
            // Find pivot
            var maxRow = col
            var maxVal = kotlin.math.abs(AA[col][col])
            for (row in col + 1 until 8) {
                if (kotlin.math.abs(AA[row][col]) > maxVal) {
                    maxVal = kotlin.math.abs(AA[row][col])
                    maxRow = row
                }
            }
            
            if (maxVal < 1e-10f) return null  // Singular matrix
            
            // Swap rows
            if (maxRow != col) {
                val tempRow = AA[col]
                AA[col] = AA[maxRow]
                AA[maxRow] = tempRow
                val tempB = b[col]
                b[col] = b[maxRow]
                b[maxRow] = tempB
            }
            
            // Eliminate
            for (row in col + 1 until 8) {
                val factor = AA[row][col] / AA[col][col]
                for (c in col until 8) {
                    AA[row][c] -= factor * AA[col][c]
                }
                b[row] -= factor * b[col]
            }
        }
        
        // Back substitution
        val x = FloatArray(8)
        for (row in 7 downTo 0) {
            var sum = b[row]
            for (col in row + 1 until 8) {
                sum -= AA[row][col] * x[col]
            }
            if (kotlin.math.abs(AA[row][row]) < 1e-10f) return null
            x[row] = sum / AA[row][row]
        }
        
        return floatArrayOf(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1f)
    }
    
    /**
     * Release resources.
     */
    fun release() {
        try {
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
