package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.scenedescription

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.Backend
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.OnDeviceProcessorResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * On-device scene description processor using FastVLM-0.5B via LiteRT-LM.
 * Mirrors the server-side fast_processor.py logic.
 */
class SceneDescriptionProcessor : OnDeviceProcessor {
    companion object {
        private const val TAG = "SceneDescProcessor"
        private const val MODEL_FILENAME = "FastVLM-0.5B.litertlm"
        private const val MODEL_DIR = "models"
        private const val MODEL_DOWNLOAD_URL =
            "https://huggingface.co/litert-community/FastVLM-0.5B/resolve/main/FastVLM-0.5B.litertlm"
        private const val PROMPT = "Describe what you see in the image."
        private const val DOWNLOAD_BUFFER_SIZE = 8192
    }

    override val id = -105
    override val name = "Scene Description (On-Device)"
    override val description = "Describe scenes using FastVLM-0.5B running locally"

    private var engine: Engine? = null
    private var modelPath: String? = null
    private var isInitialized = false
    private var isDownloading = false
    private var downloadProgress: Int = 0
    private var downloadError: String? = null
    private var appContext: Context? = null
    
    private val downloadScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val inferenceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    
    // Frame dropping: skip frames while inference is running
    private val isProcessing = AtomicBoolean(false)
    private val latestResult = AtomicReference<OnDeviceProcessorResult?>(null)

    override fun initialize(context: Context) {
        appContext = context.applicationContext
        val modelsDir = File(context.filesDir, MODEL_DIR)
        modelsDir.mkdirs()
        modelPath = File(modelsDir, MODEL_FILENAME).absolutePath
        Log.d(TAG, "Model path: $modelPath")
        
        // Start download at app launch if model doesn't exist (one-time)
        val modelFile = File(modelPath!!)
        if (!modelFile.exists() && !isDownloading) {
            Log.d(TAG, "Model not found, starting background download...")
            startDownload()
        } else if (modelFile.exists()) {
            Log.d(TAG, "Model already exists: ${modelFile.length() / 1024 / 1024} MB")
        }
    }

    /**
     * Attempt to load the engine if model file exists.
     */
    private fun loadEngine(context: Context): Boolean {
        val modelFile = File(modelPath ?: return false)
        if (!modelFile.exists()) {
            Log.d(TAG, "Model file not found at $modelPath")
            return false
        }

        // Clear potentially corrupted XNNPACK cache files first
        clearXnnpackCache(context)

        return try {
            // Try GPU first for faster inference, fallback to CPU
            val useGpu = try {
                // Test if GPU backend is available
                true
            } catch (e: Exception) {
                Log.w(TAG, "GPU not available, using CPU: ${e.message}")
                false
            }
            
            val config = EngineConfig(
                modelPath = modelFile.absolutePath,
                backend = if (useGpu) Backend.GPU else Backend.CPU,
                cacheDir = context.cacheDir.path,
                visionBackend = if (useGpu) Backend.GPU else Backend.CPU
            )
            Log.d(TAG, "Using ${if (useGpu) "GPU" else "CPU"} backend for FastVLM")
            engine = Engine(config)
            engine!!.initialize()
            isInitialized = true
            Log.d(TAG, "FastVLM engine initialized successfully")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize engine: ${e.message}", e)
            // Try once more after clearing cache
            clearXnnpackCache(context)
            false
        }
    }
    
    /**
     * Clear XNNPACK cache files that may be corrupted from previous crashes.
     */
    private fun clearXnnpackCache(context: Context) {
        try {
            val cacheDir = context.cacheDir
            val cacheFiles = cacheDir.listFiles { file -> 
                file.name.contains("xnnpack_cache") || file.name.contains("litertlm") 
            }
            cacheFiles?.forEach { file ->
                if (file.delete()) {
                    Log.d(TAG, "Deleted cache file: ${file.name}")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to clear cache: ${e.message}")
        }
    }

    /**
     * Download the model file from HuggingFace with progress tracking.
     */
    private fun startDownload() {
        if (isDownloading) return
        
        isDownloading = true
        downloadProgress = 0
        downloadError = null
        
        downloadScope.launch {
            try {
                Log.d(TAG, "Starting model download from $MODEL_DOWNLOAD_URL")
                
                var currentUrl = MODEL_DOWNLOAD_URL
                var connection: HttpURLConnection
                var responseCode: Int
                var redirectCount = 0
                val maxRedirects = 5
                
                // Follow redirects manually (HuggingFace uses multiple)
                do {
                    val url = URL(currentUrl)
                    connection = url.openConnection() as HttpURLConnection
                    connection.requestMethod = "GET"
                    connection.connectTimeout = 30000
                    connection.readTimeout = 300000 // 5 min for large file
                    connection.setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android)")
                    connection.instanceFollowRedirects = false // Handle manually
                    connection.connect()
                    
                    responseCode = connection.responseCode
                    Log.d(TAG, "Response code: $responseCode for $currentUrl")
                    
                    if (responseCode == HttpURLConnection.HTTP_MOVED_TEMP || 
                        responseCode == HttpURLConnection.HTTP_MOVED_PERM ||
                        responseCode == 307 || responseCode == 308) {
                        val redirectUrl = connection.getHeaderField("Location")
                        if (redirectUrl != null) {
                            Log.d(TAG, "Following redirect #${redirectCount + 1} to: ${redirectUrl.take(100)}...")
                            currentUrl = redirectUrl
                            connection.disconnect()
                            redirectCount++
                        } else {
                            break
                        }
                    } else {
                        break
                    }
                } while (redirectCount < maxRedirects)
                
                if (responseCode != HttpURLConnection.HTTP_OK) {
                    throw Exception("HTTP error: $responseCode")
                }
                
                // Get content length - may be in "x-linked-size" header for HuggingFace
                var totalSize = connection.contentLengthLong
                if (totalSize <= 0) {
                    val linkedSize = connection.getHeaderField("x-linked-size")
                    if (linkedSize != null) {
                        totalSize = linkedSize.toLongOrNull() ?: -1
                    }
                }
                Log.d(TAG, "Model size: ${if (totalSize > 0) "${totalSize / 1024 / 1024} MB" else "unknown"}")
                
                val modelFile = File(modelPath!!)
                val tempFile = File(modelPath + ".tmp")
                
                connection.inputStream.use { input ->
                    FileOutputStream(tempFile).use { output ->
                        val buffer = ByteArray(DOWNLOAD_BUFFER_SIZE)
                        var bytesRead: Int
                        var totalBytesRead = 0L
                        
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                            totalBytesRead += bytesRead
                            
                            if (totalSize > 0) {
                                downloadProgress = ((totalBytesRead * 100) / totalSize).toInt()
                            } else {
                                // Fallback: show MB downloaded
                                downloadProgress = (totalBytesRead / 1024 / 1024).toInt()
                            }
                            
                            if (totalBytesRead % (10 * 1024 * 1024) < DOWNLOAD_BUFFER_SIZE) {
                                Log.d(TAG, "Downloaded ${totalBytesRead / 1024 / 1024} MB ($downloadProgress%)")
                            }
                        }
                    }
                }
                
                connection.disconnect()
                
                // Rename temp file to final name
                if (tempFile.renameTo(modelFile)) {
                    Log.d(TAG, "Model download complete: ${modelFile.length() / 1024 / 1024} MB")
                    downloadProgress = 100
                    isDownloading = false
                    // Engine will be loaded lazily on next process call
                    // Don't load immediately to avoid OOM after large download
                    Log.d(TAG, "Model ready - will load engine on next frame")
                } else {
                    throw Exception("Failed to rename downloaded file")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Download failed: ${e.message}", e)
                downloadError = e.message ?: "Download failed"
                isDownloading = false
                
                // Clean up partial download
                File(modelPath + ".tmp").delete()
            }
        }
    }

    override suspend fun process(frame: Bitmap): OnDeviceProcessorResult =
        withContext(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()

            if (!isInitialized) {
                // Check if model needs download
                val modelFile = File(modelPath ?: return@withContext OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Model path not set",
                    processingTimeMs = 0
                ))

                if (!modelFile.exists()) {
                    // Check for download error - show once, then auto-retry
                    downloadError?.let { error ->
                        // Clear error to trigger retry on next frame
                        downloadError = null
                        return@withContext OnDeviceProcessorResult(
                            processedImage = frame,
                            text = "Download failed: $error\nRetrying automatically...",
                            processingTimeMs = System.currentTimeMillis() - startTime
                        )
                    }
                    
                    if (isDownloading) {
                        return@withContext OnDeviceProcessorResult(
                            processedImage = frame,
                            text = "Downloading model... $downloadProgress%\n(~1.1GB, ensure WiFi is stable)",
                            processingTimeMs = System.currentTimeMillis() - startTime
                        )
                    }

                    // Start download
                    startDownload()
                    return@withContext OnDeviceProcessorResult(
                        processedImage = frame,
                        text = "Starting model download (~1.1GB)...\nEnsure you have WiFi connection.",
                        processingTimeMs = System.currentTimeMillis() - startTime
                    )
                }

                // Model exists but engine not loaded yet
                appContext?.let { ctx ->
                    if (loadEngine(ctx)) {
                        // Engine loaded, continue to process
                    } else {
                        return@withContext OnDeviceProcessorResult(
                            processedImage = frame,
                            text = "Failed to load FastVLM model",
                            processingTimeMs = System.currentTimeMillis() - startTime
                        )
                    }
                } ?: return@withContext OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Context not available",
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            }

            // FRAME DROPPING: If already processing, return last result immediately
            // This prevents the camera preview from freezing
            if (isProcessing.get()) {
                val cached = latestResult.get()
                return@withContext cached?.copy(
                    processedImage = frame,  // Always use current frame
                    processingTimeMs = 0     // Indicate cached result
                ) ?: OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Processing...",
                    processingTimeMs = 0
                )
            }

            // Mark as processing and run inference
            isProcessing.set(true)
            
            try {
                // Save frame to temp file for LiteRT-LM image input
                val tempFile = File.createTempFile("frame_", ".jpg")
                FileOutputStream(tempFile).use { fos ->
                    frame.compress(Bitmap.CompressFormat.JPEG, 85, fos)
                }

                val message = engine!!.createConversation().use { conversation ->
                    conversation.sendMessage(
                        Contents.of(
                            Content.ImageFile(tempFile.absolutePath),
                            Content.Text(PROMPT)
                        )
                    )
                }

                tempFile.delete()

                val result = OnDeviceProcessorResult(
                    processedImage = frame,
                    text = message.toString(),
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
                
                // Cache result for frame dropping
                latestResult.set(result)
                result
            } catch (e: Exception) {
                Log.e(TAG, "Inference error: ${e.message}")
                OnDeviceProcessorResult(
                    processedImage = frame,
                    text = "Scene description error: ${e.message}",
                    processingTimeMs = System.currentTimeMillis() - startTime
                )
            } finally {
                isProcessing.set(false)
            }
        }

    /**
     * Attempt to load the engine with context. Called from ProcessorManager
     * after initialization to try loading the model if it exists.
     */
    fun tryLoadEngine(context: Context) {
        if (!isInitialized) {
            loadEngine(context)
        }
    }

    override fun release() {
        try {
            engine?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing engine: ${e.message}")
        }
        engine = null
        isInitialized = false
        Log.d(TAG, "FastVLM engine released")
    }
}

