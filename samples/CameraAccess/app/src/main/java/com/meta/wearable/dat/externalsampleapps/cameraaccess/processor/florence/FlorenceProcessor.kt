package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.florence

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Base64
import android.util.Log
import kotlin.math.ceil
import android.webkit.ConsoleMessage
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.ByteArrayOutputStream
import java.io.InputStreamReader
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean  // modelReady
import java.util.concurrent.atomic.AtomicReference

/**
 * On-device dense region captioning using Florence-2 via a headless WebView.
 *
 * ## Architecture
 * A hidden WebView loads `florence/florence_inference.html` from assets. That page:
 *   1. Fetches Transformers.js from jsDelivr CDN (cached after first load, ~1.5 MB).
 *   2. Loads the Florence-2 ONNX model files via a loopback HTTP server ([AssetServer])
 *      that streams them straight from APK assets — no network needed.
 *   3. Runs on WebGPU if available, falls back to WASM.
 *
 * ## Why a loopback server instead of shouldInterceptRequest
 * Android WebView's shouldInterceptRequest is NOT called for fetch() API requests
 * initiated from cross-origin ES modules (e.g., Transformers.js loaded from CDN).
 * A real TCP server on localhost is always reachable by any fetch() call.
 *
 * ## Model files (assets/models/florence/)
 *   vision_encoder_int8.onnx         — DaViT image encoder
 *   decoder_model_merged_int8.onnx   — BART text encoder + first-token decoder
 *   decoder_with_past_model_int8.onnx — BART decoder with KV cache
 *   config.json / tokenizer*.json / preprocessor_config.json / generation_config.json
 *
 * ## URL rewriting
 * Transformers.js requests ONNX files at `…/florence/onnx/<file>.onnx`
 * (inside an `onnx/` subfolder). [AssetServer] strips `/onnx/` so the assets stay
 * in their current flat layout without any directory restructuring.
 *
 * ## Threading
 *  - [initialize] must be called on the main thread (WebView creation requirement).
 *  - [process] is called on Dispatchers.Default; it switches to Main only for the
 *    [WebView.evaluateJavascript] call, then suspends until the JS callback fires.
 *  - [@JavascriptInterface] callbacks run on the WebView binder thread; they simply
 *    complete a [CompletableDeferred], which is thread-safe.
 */
class FlorenceProcessor : OnDeviceProcessor() {

    companion object {
        private const val TAG = "FlorenceProcessor"
        private const val INFERENCE_TIMEOUT_MS = 60_000L
        private const val JPEG_QUALITY = 85
        private const val MAX_ANALYZE_PX = 1024  // max dimension before downsampling for analyzeRegions()
        private val COLORS = intArrayOf(
            Color.rgb(255, 82,  82),  // red
            Color.rgb(0,   200, 83),  // green
            Color.rgb(41,  182, 246), // blue
            Color.rgb(255, 193, 7),   // amber
            Color.rgb(171, 71,  188), // purple
            Color.rgb(0,   188, 212), // cyan
            Color.rgb(255, 138, 101), // orange
            Color.rgb(105, 240, 174), // mint
        )
    }

    override val id          = -109
    override val name        = "Dense Captions (Florence-2)"
    override val description = "Florence-2 dense region captioning via WebGPU/WASM"

    // ── State ─────────────────────────────────────────────────────────────────
    private var webView: WebView? = null
    private var assetServer: AssetServer? = null
    private val modelReady    = AtomicBoolean(false)
    private val loadingStatus = AtomicReference("Loading Florence-2…")

    /** Completed by the JS bridge when an inference result arrives. */
    @Volatile private var pendingResult: CompletableDeferred<String?>? = null

    // ── Paints ────────────────────────────────────────────────────────────────
    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }
    private val labelBgPaint = Paint().apply { style = Paint.Style.FILL }
    private val labelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 34f
        isFakeBoldText = true
        isAntiAlias = true
    }
    private val statusPaint = Paint().apply {
        color = Color.WHITE
        textSize = 38f
        isAntiAlias = true
        isFakeBoldText = true
        setShadowLayer(3f, 1f, 1f, Color.BLACK)
    }

    // ── JavaScript bridge (called from WebView binder thread) ─────────────────

    @JavascriptInterface
    fun onModelReady() {
        modelReady.set(true)
        loadingStatus.set("")
        Log.d(TAG, "Florence-2 model ready")
    }

    @JavascriptInterface
    fun onModelProgress(message: String) {
        loadingStatus.set(message)
        Log.d(TAG, "Florence progress: $message")
    }

    @JavascriptInterface
    fun onInferenceResult(json: String) {
        pendingResult?.complete(json)
        // Log the full result up to 500 chars so we can see if bboxes/labels are actually populated.
        Log.d(TAG, "Inference result (${json.length} chars): ${json.take(500)}")
    }

    @JavascriptInterface
    fun onError(message: String) {
        pendingResult?.complete(null)
        Log.e(TAG, "Florence JS error: $message")
        loadingStatus.set("Error: $message")
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    @SuppressLint("SetJavaScriptEnabled")
    override fun initialize(context: Context) {
        // Must be called on the main thread.
        val ctx = context.applicationContext

        // Start the loopback asset server on an OS-assigned free port.
        val server = AssetServer(ctx.assets)
        assetServer = server
        val port = server.port
        Log.d(TAG, "Asset server started on port $port")

        // Pre-compute the HTML with the localhost model URL injected so that
        // Transformers.js fetches model files from our server.
        // Rewrite env.localModelPath in the HTML from the placeholder URL to our
        // loopback server.  The model name ('florence') stays as-is; Transformers.js
        // appends it to localModelPath when constructing each file URL.
        val htmlBytes = ctx.assets.open("florence/florence_inference.html")
            .bufferedReader().readText()
            .replace(
                "https://appassets.androidplatform.net/assets/models/",
                "http://localhost:$port/models/"
            )
            .toByteArray(Charsets.UTF_8)

        // Register the HTML as a virtual file so the server can serve it.
        // This must happen before loadUrl so the server is ready when the browser asks.
        server.setVirtualFile("florence/florence_inference.html", htmlBytes)

        webView = WebView(ctx).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true   // required by Transformers.js
            settings.allowContentAccess = true

            // Register the Kotlin bridge under the name used in the HTML.
            addJavascriptInterface(this@FlorenceProcessor, "AndroidBridge")

            webChromeClient = object : WebChromeClient() {
                override fun onConsoleMessage(msg: ConsoleMessage): Boolean {
                    val level = when (msg.messageLevel()) {
                        ConsoleMessage.MessageLevel.ERROR   -> Log.ERROR
                        ConsoleMessage.MessageLevel.WARNING -> Log.WARN
                        else                               -> Log.DEBUG
                    }
                    Log.println(level, "$TAG/JS", "${msg.message()} [${msg.sourceId()}:${msg.lineNumber()}]")
                    return true
                }
            }

            webViewClient = object : WebViewClient() {
                @Suppress("OVERRIDE_DEPRECATION")
                override fun onReceivedError(
                    view: WebView,
                    errorCode: Int,
                    description: String?,
                    failingUrl: String?
                ) {
                    Log.e(TAG, "WebView error $errorCode: $description @ $failingUrl")
                }
            }

            // Load the HTML via a genuine HTTP request to localhost so the page
            // origin is http://localhost — not a null/data: origin (which Chrome
            // blocks from accessing localhost via fetch()).
            loadUrl("http://localhost:$port/florence/florence_inference.html")
        }

        Log.d(TAG, "FlorenceProcessor initialized — serving assets on port $port")
    }

    override suspend fun process(frame: Bitmap): OnDeviceProcessorResult {
        val t0 = System.currentTimeMillis()

        // Not ready yet — show loading status on the live frame.
        if (!modelReady.get()) {
            return OnDeviceProcessorResult(
                processedImage = stampStatus(frame, loadingStatus.get()),
                text           = loadingStatus.get(),
                processingTimeMs = 0
            )
        }

        // Frame-drop guard: only one inference at a time.
        if (!isProcessing.compareAndSet(false, true)) {
            return OnDeviceProcessorResult(
                processedImage = frame,
                text           = null,
                processingTimeMs = 0
            )
        }

        return withContext(Dispatchers.Default) {
            try {
                Log.d(TAG, "process(): frame ${frame.width}×${frame.height}")
                val b64 = encodeFrame(frame)
                Log.d(TAG, "process(): encoded ${b64.length} base64 chars  prefix='${b64.take(8)}'  — calling runInference")
                val json = runInferenceRaw("data:image/jpeg;base64,$b64", "<DENSE_REGION_CAPTION>")

                if (json == null) {
                    Log.w(TAG, "process(): inference timed out after ${INFERENCE_TIMEOUT_MS / 1000}s")
                    return@withContext OnDeviceProcessorResult(
                        processedImage = stampStatus(frame, "Florence: inference timed out"),
                        text           = "Timeout after ${INFERENCE_TIMEOUT_MS / 1000}s",
                        processingTimeMs = System.currentTimeMillis() - t0
                    )
                }

                val regions = parseResult(json)
                Log.d(TAG, "process(): ${regions.size} region(s) in ${System.currentTimeMillis() - t0}ms")
                val annotated = drawRegions(frame, regions)
                val summary = regions.joinToString(", ") { it.label.take(25) }
                    .ifEmpty { "No regions detected" }

                OnDeviceProcessorResult(
                    processedImage   = annotated,
                    text             = summary,
                    processingTimeMs = System.currentTimeMillis() - t0
                )
            } catch (e: Exception) {
                Log.e(TAG, "process() error: ${e.message}", e)
                OnDeviceProcessorResult(
                    processedImage   = frame,
                    text             = "Florence error: ${e.message}",
                    processingTimeMs = System.currentTimeMillis() - t0
                )
            } finally {
                isProcessing.set(false)
            }
        }
    }

    override fun release() {
        webView?.destroy()
        webView = null
        assetServer?.stop()
        assetServer = null
        modelReady.set(false)
        Log.d(TAG, "FlorenceProcessor released")
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /** Encode [bitmap] as a base64 JPEG string for WebView transfer. */
    private fun encodeFrame(bitmap: Bitmap): String {
        val baos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, baos)
        return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP)
    }

    /**
     * Core inference call. Sends [imageDataUrl] to the WebView with the given
     * [task] token and optional [extraPrompt], then suspends until the JS bridge
     * fires [onInferenceResult] or the timeout elapses.
     *
     * Returns the raw JSON string or null on timeout/error.
     * Caller is responsible for the [isProcessing] guard.
     */
    private suspend fun runInferenceRaw(
        imageDataUrl: String,
        task: String,
        extraPrompt: String? = null,
    ): String? {
        val deferred = CompletableDeferred<String?>()
        pendingResult = deferred

        val extraArg = if (extraPrompt != null) ", '${extraPrompt.replace("'", "\\'")}'" else ", null"
        withContext(Dispatchers.Main) {
            webView?.evaluateJavascript(
                "window.runInference('$imageDataUrl', '$task'$extraArg)",
                null
            )
        }
        return withTimeoutOrNull(INFERENCE_TIMEOUT_MS) { deferred.await() }
    }

    // ── Result parsers ────────────────────────────────────────────────────────

    /**
     * Parse bbox-style response:
     * `{ "<TASK>": { "bboxes": [[x1,y1,x2,y2],…], "labels": ["…",…] } }`
     */
    private fun parseBboxResult(json: String, task: String): List<RegionCaption> {
        return try {
            val taskResult = JSONObject(json).getJSONObject(task)
            val bboxArray  = taskResult.getJSONArray("bboxes")
            val labelArray = taskResult.getJSONArray("labels")
            val count      = minOf(bboxArray.length(), labelArray.length())
            (0 until count).mapNotNull { i ->
                val b = bboxArray.get(i) as? JSONArray ?: return@mapNotNull null
                if (b.length() < 4) return@mapNotNull null
                RegionCaption(
                    bbox  = RectF(
                        b.getDouble(0).toFloat(), b.getDouble(1).toFloat(),
                        b.getDouble(2).toFloat(), b.getDouble(3).toFloat(),
                    ),
                    label = labelArray.getString(i),
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "parseBboxResult($task) error: ${e.message} — raw: ${json.take(200)}")
            emptyList()
        }
    }

    /**
     * Parse segmentation response:
     * `{ "<TASK>": { "polygons": [[[x,y],…],…], "labels": ["…",…] } }`
     *
     * Each polygon is returned as a flat FloatArray `[x0,y0, x1,y1, …]`.
     */
    private fun parseSegmentationResult(json: String, task: String): List<Pair<FloatArray, String>> {
        return try {
            val taskResult   = JSONObject(json).getJSONObject(task)
            val polygonsArr  = taskResult.getJSONArray("polygons")
            val labelArray   = taskResult.getJSONArray("labels")
            val count        = minOf(polygonsArr.length(), labelArray.length())
            (0 until count).mapNotNull { i ->
                val polyGroup = polygonsArr.get(i) as? JSONArray ?: return@mapNotNull null
                if (polyGroup.length() == 0) return@mapNotNull null
                // Normalise nesting:
                //   polygons[i] = [[x,y],[x,y],…]           → pointsArr = polyGroup
                //   polygons[i] = [[[x,y],[x,y],…]]         → pointsArr = polyGroup[0]
                //   polygons[i] = [[[[x,y],…]]]  (rare)     → pointsArr = polyGroup[0][0]
                val first = polyGroup.get(0) as? JSONArray ?: return@mapNotNull null
                val pointsArr: JSONArray = if (first.length() > 0 && first.get(0) is JSONArray) {
                    // polyGroup[0] is itself an array of points → polyGroup is the wrapper
                    val second = first.get(0) as? JSONArray ?: return@mapNotNull null
                    if (second.length() > 0 && second.get(0) is JSONArray) second.get(0) as JSONArray
                    else first
                } else {
                    // polyGroup[0] is a [x, y] pair → polyGroup is the points array
                    polyGroup
                }
                val pts = FloatArray(pointsArr.length() * 2)
                for (j in 0 until pointsArr.length()) {
                    val pt = pointsArr.getJSONArray(j)
                    pts[j * 2]     = pt.getDouble(0).toFloat()
                    pts[j * 2 + 1] = pt.getDouble(1).toFloat()
                }
                pts to labelArray.getString(i)
            }
        } catch (e: Exception) {
            Log.e(TAG, "parseSegmentationResult($task) error: ${e.message} — raw: ${json.take(200)}")
            emptyList()
        }
    }

    /**
     * Parse plain-string response:
     * `{ "<TASK>": "some text" }`
     */
    private fun parseStringResult(json: String, task: String): String? {
        return try {
            JSONObject(json).optString(task).ifEmpty { null }
        } catch (e: Exception) {
            Log.e(TAG, "parseStringResult($task) error: ${e.message}")
            null
        }
    }

    // Kept for backward compat with process() and analyzeRegions() callers.
    private fun parseResult(json: String): List<RegionCaption> =
        parseBboxResult(json, "<DENSE_REGION_CAPTION>")

    /** Draw region bounding boxes + labels onto a mutable copy of [frame]. */
    private fun drawRegions(frame: Bitmap, regions: List<RegionCaption>): Bitmap {
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        for ((idx, region) in regions.withIndex()) {
            val color = COLORS[idx % COLORS.size]
            boxPaint.color = color
            labelBgPaint.color = color

            canvas.drawRect(region.bbox, boxPaint)

            val label     = region.label.take(40)
            val textWidth = labelPaint.measureText(label)
            val bgTop     = (region.bbox.top - 44f).coerceAtLeast(0f)
            canvas.drawRect(
                region.bbox.left,
                bgTop,
                region.bbox.left + textWidth + 12f,
                bgTop + 44f,
                labelBgPaint
            )
            canvas.drawText(label, region.bbox.left + 6f, bgTop + 32f, labelPaint)
        }

        return output
    }

    /** Stamp a status string onto a mutable copy of [frame] (used while loading). */
    private fun stampStatus(frame: Bitmap, status: String): Bitmap {
        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        Canvas(output).drawText(status, 20f, 60f, statusPaint)
        return output
    }

    /**
     * Run Florence-2 dense region captioning on [bitmap] and return raw regions
     * as (bbox-in-original-bitmap-px, label) pairs.
     *
     * The bitmap is downsampled to at most [MAX_ANALYZE_PX] on its longest side
     * before encoding to keep the base64 payload small; returned bboxes are
     * scaled back to the original bitmap's pixel space.
     *
     * Returns an empty list if the model isn't ready, another inference is already
     * running, or the inference fails / times out.
     */
    suspend fun analyzeRegions(bitmap: Bitmap): List<Pair<RectF, String>> {
        if (!modelReady.get()) return emptyList()
        if (!isProcessing.compareAndSet(false, true)) return emptyList()

        return withContext(Dispatchers.Default) {
            try {
                Log.d(TAG, "analyzeRegions(): input ${bitmap.width}×${bitmap.height}")
                // Downsample to avoid sending a huge payload to the WebView.
                val maxPx = MAX_ANALYZE_PX
                val scaleFactor = if (bitmap.width > maxPx || bitmap.height > maxPx) {
                    minOf(maxPx.toFloat() / bitmap.width, maxPx.toFloat() / bitmap.height)
                } else 1f

                val scaled = if (scaleFactor < 1f) {
                    Bitmap.createScaledBitmap(
                        bitmap,
                        (bitmap.width  * scaleFactor).toInt().coerceAtLeast(1),
                        (bitmap.height * scaleFactor).toInt().coerceAtLeast(1),
                        true
                    )
                } else bitmap

                if (scaleFactor < 1f) {
                    Log.d(TAG, "analyzeRegions(): downsampled to ${scaled.width}×${scaled.height} (scale=${"%.3f".format(scaleFactor)})")
                }
                val b64 = encodeFrame(scaled)
                Log.d(TAG, "analyzeRegions(): encoded ${b64.length} base64 chars — calling runInference")
                if (scaleFactor < 1f) scaled.recycle()

                val json = runInferenceRaw("data:image/jpeg;base64,$b64", "<DENSE_REGION_CAPTION>")
                if (json == null) {
                    Log.w(TAG, "analyzeRegions(): inference timed out after ${INFERENCE_TIMEOUT_MS / 1000}s")
                    return@withContext emptyList()
                }

                // Scale bboxes from downsampled-image space back to original bitmap space.
                val scaleX = if (scaleFactor < 1f) 1f / scaleFactor else 1f
                val scaleY = scaleX
                val captions = parseResult(json)
                Log.d(TAG, "analyzeRegions(): ${captions.size} region(s) parsed")
                captions.map { rc ->
                    val bbox = if (scaleFactor < 1f) {
                        RectF(rc.bbox.left * scaleX, rc.bbox.top * scaleY,
                              rc.bbox.right * scaleX, rc.bbox.bottom * scaleY)
                    } else rc.bbox
                    bbox to rc.label
                }
            } catch (e: Exception) {
                Log.e(TAG, "analyzeRegions() error: ${e.message}", e)
                emptyList()
            } finally {
                isProcessing.set(false)
            }
        }
    }

    /**
     * Segment a specific region of [bitmap] defined by [bbox] (pixels in [bitmap]'s space).
     *
     * Sends `<REGION_TO_SEGMENTATION>` with the bbox encoded as Florence's loc-token prompt.
     * Returns a list of (polygon as flat [x0,y0,x1,y1,…] in [bitmap]'s pixel space, label).
     * Returns empty if not ready, another inference is running, or the model finds nothing.
     */
    suspend fun segmentRegion(bitmap: Bitmap, bbox: RectF): List<Pair<FloatArray, String>> {
        if (!modelReady.get()) return emptyList()
        if (!isProcessing.compareAndSet(false, true)) return emptyList()

        return withContext(Dispatchers.Default) {
            var crop: Bitmap? = null
            var scaled: Bitmap? = null
            try {
                // Crop to the bbox + 30 % padding on each side so Florence sees the
                // object in context but at a zoom level where it fills most of the image.
                // Running on the full (panorama-wide) bitmap with small loc tokens
                // consistently fails — Florence needs the object to be prominent.
                val padFrac   = 0.3f
                val cropLeft  = (bbox.left   - bbox.width()  * padFrac).coerceAtLeast(0f).toInt()
                val cropTop   = (bbox.top    - bbox.height() * padFrac).coerceAtLeast(0f).toInt()
                val cropRight = (bbox.right  + bbox.width()  * padFrac).coerceAtMost(bitmap.width.toFloat()).toInt()
                val cropBot   = (bbox.bottom + bbox.height() * padFrac).coerceAtMost(bitmap.height.toFloat()).toInt()
                val cropW     = (cropRight - cropLeft).coerceAtLeast(1)
                val cropH     = (cropBot   - cropTop ).coerceAtLeast(1)
                crop = Bitmap.createBitmap(bitmap, cropLeft, cropTop, cropW, cropH)

                // Scale the crop if it exceeds MAX_ANALYZE_PX.
                val maxPx = MAX_ANALYZE_PX
                val scale = if (cropW > maxPx || cropH > maxPx)
                    minOf(maxPx.toFloat() / cropW, maxPx.toFloat() / cropH) else 1f
                scaled = if (scale < 1f)
                    Bitmap.createScaledBitmap(crop,
                        (cropW * scale).toInt().coerceAtLeast(1),
                        (cropH * scale).toInt().coerceAtLeast(1), true)
                else crop

                // Loc tokens for the tight bbox within the scaled crop (0–999 range).
                val sw = scaled.width.toFloat(); val sh = scaled.height.toFloat()
                val lx1 = ((bbox.left   - cropLeft) * scale / sw * 999).toInt().coerceIn(0, 999)
                val ly1 = ((bbox.top    - cropTop ) * scale / sh * 999).toInt().coerceIn(0, 999)
                val lx2 = ((bbox.right  - cropLeft) * scale / sw * 999).toInt().coerceIn(0, 999)
                val ly2 = ((bbox.bottom - cropTop ) * scale / sh * 999).toInt().coerceIn(0, 999)
                val regionPrompt = "<loc_$lx1><loc_$ly1><loc_$lx2><loc_$ly2>"
                Log.d(TAG, "segmentRegion: crop=${cropW}×${cropH} scaled=${scaled.width}×${scaled.height} prompt=$regionPrompt")

                val b64  = encodeFrame(scaled)
                val json = runInferenceRaw("data:image/jpeg;base64,$b64",
                    "<REGION_TO_SEGMENTATION>", regionPrompt)
                if (json == null) {
                    Log.w(TAG, "segmentRegion: timed out"); return@withContext emptyList()
                }
                Log.d(TAG, "segmentRegion raw json: ${json.take(400)}")

                // Polygon coords from Florence are in scaled-crop pixel space.
                // Translate back to original bitmap pixel space.
                val raw = parseSegmentationResult(json, "<REGION_TO_SEGMENTATION>")
                raw.map { (pts, label) ->
                    FloatArray(pts.size) { j ->
                        if (j % 2 == 0) pts[j] / scale + cropLeft
                        else            pts[j] / scale + cropTop
                    } to label
                }
            } catch (e: Exception) {
                Log.e(TAG, "segmentRegion error: ${e.message}", e)
                emptyList()
            } finally {
                if (scaled !== crop) scaled?.recycle()
                crop?.recycle()
                isProcessing.set(false)
            }
        }
    }

    /**
     * Generate a detailed description for the region of [bitmap] defined by [bbox].
     *
     * Uses `<REGION_TO_DESCRIPTION>`. Returns null if not ready or inference fails.
     */
    suspend fun describeRegion(bitmap: Bitmap, bbox: RectF): String? {
        if (!modelReady.get()) return null
        if (!isProcessing.compareAndSet(false, true)) return null

        return withContext(Dispatchers.Default) {
            try {
                val maxPx = MAX_ANALYZE_PX
                val scale = if (bitmap.width > maxPx || bitmap.height > maxPx)
                    minOf(maxPx.toFloat() / bitmap.width, maxPx.toFloat() / bitmap.height) else 1f
                val scaled = if (scale < 1f)
                    Bitmap.createScaledBitmap(bitmap,
                        (bitmap.width * scale).toInt().coerceAtLeast(1),
                        (bitmap.height * scale).toInt().coerceAtLeast(1), true)
                else bitmap

                val sw = scaled.width.toFloat(); val sh = scaled.height.toFloat()
                val x1 = ((bbox.left  * scale / sw) * 999).toInt().coerceIn(0, 999)
                val y1 = ((bbox.top   * scale / sh) * 999).toInt().coerceIn(0, 999)
                val x2 = ((bbox.right * scale / sw) * 999).toInt().coerceIn(0, 999)
                val y2 = ((bbox.bottom* scale / sh) * 999).toInt().coerceIn(0, 999)
                val regionPrompt = "<loc_$x1><loc_$y1><loc_$x2><loc_$y2>"

                val b64 = encodeFrame(scaled)
                if (scale < 1f) scaled.recycle()

                val json = runInferenceRaw("data:image/jpeg;base64,$b64",
                    "<REGION_TO_DESCRIPTION>", regionPrompt)
                if (json == null) { Log.w(TAG, "describeRegion: timed out"); return@withContext null }
                parseStringResult(json, "<REGION_TO_DESCRIPTION>")
            } catch (e: Exception) {
                Log.e(TAG, "describeRegion error: ${e.message}", e)
                null
            } finally {
                isProcessing.set(false)
            }
        }
    }

    /**
     * Generate a short or detailed caption for the whole [bitmap].
     *
     * @param detailed  false → `<CAPTION>`, true → `<MORE_DETAILED_CAPTION>`
     */
    suspend fun captionImage(bitmap: Bitmap, detailed: Boolean = false): String? {
        if (!modelReady.get()) return null
        if (!isProcessing.compareAndSet(false, true)) return null

        return withContext(Dispatchers.Default) {
            try {
                val task = if (detailed) "<MORE_DETAILED_CAPTION>" else "<CAPTION>"
                val b64  = encodeFrame(bitmap)
                val json = runInferenceRaw("data:image/jpeg;base64,$b64", task)
                if (json == null) { Log.w(TAG, "captionImage: timed out"); return@withContext null }
                parseStringResult(json, task)
            } catch (e: Exception) {
                Log.e(TAG, "captionImage error: ${e.message}", e)
                null
            } finally {
                isProcessing.set(false)
            }
        }
    }

    /**
     * Tiled variant of [analyzeRegions] for wide-aspect bitmaps (e.g. stitched panoramas).
     *
     * Splits [bitmap] into vertical strips of width = [bitmap.height] (square tiles),
     * runs [analyzeRegions] on each in sequence, then offsets each bbox's X by the
     * tile's left edge. The last tile may be narrower than tall; Florence handles it fine.
     *
     * Returns all regions in [bitmap]'s original pixel coordinate space.
     * Y coordinates are meaningful because each tile has a ~1:1 aspect ratio,
     * so Florence's internal padding is minimal.
     */
    suspend fun analyzeRegionsTiled(bitmap: Bitmap): List<Pair<RectF, String>> {
        val tileW   = bitmap.height  // square tiles
        val numTiles = ceil(bitmap.width.toFloat() / tileW).toInt().coerceAtLeast(1)
        val results  = mutableListOf<Pair<RectF, String>>()

        for (i in 0 until numTiles) {
            val xStart      = i * tileW
            val actualWidth = minOf(tileW, bitmap.width - xStart)
            val tile = Bitmap.createBitmap(bitmap, xStart, 0, actualWidth, bitmap.height)
            Log.d(TAG, "analyzeRegionsTiled: tile $i/$numTiles x=[$xStart,${xStart + actualWidth}] ${actualWidth}×${bitmap.height}")

            val regions = analyzeRegions(tile)
            tile.recycle()

            for ((bbox, label) in regions) {
                results.add(RectF(bbox.left + xStart, bbox.top,
                                  bbox.right + xStart, bbox.bottom) to label)
            }
        }

        Log.d(TAG, "analyzeRegionsTiled: ${results.size} total regions across $numTiles tiles")
        return results
    }

    // ── Data ──────────────────────────────────────────────────────────────────

    private data class RegionCaption(val bbox: RectF, val label: String)

    // ── Asset HTTP server ─────────────────────────────────────────────────────

    /**
     * Minimal single-threaded-accept, per-request-thread HTTP/1.1 server that
     * streams files from APK assets. Runs on a daemon thread so it dies with the
     * process without explicit cleanup (though [stop] is still called in [release]).
     *
     * The `/onnx/` segment is stripped from ONNX file paths so that Transformers.js's
     * constructed URL (`…/florence/onnx/vision_encoder_int8.onnx`) maps to the flat
     * asset layout (`models/florence/vision_encoder_int8.onnx`).
     */
    private class AssetServer(
        private val assets: android.content.res.AssetManager
    ) {
        private val serverSocket = ServerSocket(0)  // OS picks a free port
        val port: Int get() = serverSocket.localPort

        /** In-memory files (e.g. the HTML with the port substituted in). */
        private val virtualFiles = ConcurrentHashMap<String, ByteArray>()

        fun setVirtualFile(path: String, content: ByteArray) {
            virtualFiles[path] = content
            Log.d(TAG, "AssetServer: registered virtual file '$path' (${content.size} bytes)")
        }

        init {
            Thread(
                {
                    var connId = 0
                    while (!serverSocket.isClosed) {
                        try {
                            val client = serverSocket.accept()
                            val id = ++connId
                            Thread { serve(client, id) }.start()
                        } catch (e: Exception) {
                            if (!serverSocket.isClosed)
                                Log.e(TAG, "AssetServer: accept error: ${e.message}")
                        }
                    }
                    Log.d(TAG, "AssetServer: accept loop exited")
                },
                "FlorenceAssetServer"
            ).also { it.isDaemon = true }.start()
        }

        private fun serve(socket: Socket, connId: Int) {
            socket.use {
                try {
                    val reader = BufferedReader(
                        InputStreamReader(it.getInputStream(), Charsets.ISO_8859_1)
                    )
                    val requestLine = reader.readLine()
                    if (requestLine == null) {
                        Log.w(TAG, "AssetServer #$connId: null request line — client closed without sending")
                        return
                    }
                    val parts = requestLine.split(" ")
                    val method  = parts.getOrElse(0) { "GET" }
                    val rawPath = parts.getOrElse(1) { "/" }
                        .removePrefix("/")
                        .substringBefore("?")
                        // Transformers.js requests ONNX files in an onnx/ sub-path;
                        // our assets are flat — strip the extra segment.
                        .replace("models/florence/onnx/", "models/florence/")

                    // Drain HTTP request headers (stop at the blank line).
                    while (reader.readLine()?.isNotEmpty() == true) { /* consume */ }

                    Log.d(TAG, "AssetServer #$connId: $method '$rawPath'")

                    val sendBody = (method != "HEAD")
                    val out = it.getOutputStream()

                    val virtual = virtualFiles[rawPath]
                    if (virtual != null) {
                        val mime = mimeFor(rawPath)
                        Log.d(TAG, "AssetServer #$connId: 200 (virtual ${virtual.size}B) $rawPath")
                        val header = "HTTP/1.1 200 OK\r\nContent-Type: $mime\r\n" +
                            "Content-Length: ${virtual.size}\r\n" +
                            "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
                        out.write(header.toByteArray(Charsets.ISO_8859_1))
                        if (sendBody) out.write(virtual)
                    } else {
                        serveAsset(rawPath, sendBody, out, connId)
                    }
                    out.flush()
                } catch (e: Exception) {
                    Log.e(TAG, "AssetServer #$connId: serve error: ${e.message}", e)
                }
            }
        }

        /**
         * Serve a file from APK assets with a correct Content-Length header.
         *
         * For noCompress assets (ONNX files declared in aaptOptions) we use
         * [android.content.res.AssetManager.openFd] which gives us the file length
         * directly from the APK ZIP entry without reading all bytes into memory.
         * Compressed assets (JSON config files) fall back to a full read.
         */
        private fun serveAsset(rawPath: String, sendBody: Boolean, out: java.io.OutputStream, connId: Int) {
            val mime = mimeFor(rawPath)

            // Fast path: non-compressed asset — get size from file descriptor.
            try {
                val fd = assets.openFd(rawPath)
                val length = fd.length
                Log.d(TAG, "AssetServer #$connId: 200 (fd ${length}B) $rawPath")
                val header = "HTTP/1.1 200 OK\r\nContent-Type: $mime\r\n" +
                    "Content-Length: $length\r\n" +
                    "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
                out.write(header.toByteArray(Charsets.ISO_8859_1))
                if (sendBody) fd.createInputStream().use { s -> s.copyTo(out) }
                fd.close()
                return
            } catch (_: Exception) {
                // Asset is stored compressed in the APK — openFd() fails; fall through.
            }

            // Slow path: read compressed asset into memory so we know the length.
            try {
                val bytes = assets.open(rawPath).use { it.readBytes() }
                Log.d(TAG, "AssetServer #$connId: 200 (buf ${bytes.size}B) $rawPath")
                val header = "HTTP/1.1 200 OK\r\nContent-Type: $mime\r\n" +
                    "Content-Length: ${bytes.size}\r\n" +
                    "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
                out.write(header.toByteArray(Charsets.ISO_8859_1))
                if (sendBody) out.write(bytes)
            } catch (e: Exception) {
                Log.e(TAG, "AssetServer #$connId: 404 '$rawPath' — ${e.message}")
                val notFound = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                out.write(notFound.toByteArray(Charsets.ISO_8859_1))
            }
        }

        private fun mimeFor(path: String) = when {
            path.endsWith(".json") -> "application/json"
            path.endsWith(".html") -> "text/html"
            path.endsWith(".onnx") -> "application/octet-stream"
            path.endsWith(".js")   -> "application/javascript"
            path.endsWith(".txt")  -> "text/plain"
            else                   -> "application/octet-stream"
        }

        fun stop() = runCatching { serverSocket.close() }
    }
}
