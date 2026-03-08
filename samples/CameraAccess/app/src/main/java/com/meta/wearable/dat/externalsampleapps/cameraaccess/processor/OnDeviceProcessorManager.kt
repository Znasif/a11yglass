package com.meta.wearable.dat.externalsampleapps.cameraaccess.processor

import android.content.Context
import android.util.Log
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ProcessorInfo
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.basic.BasicProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.fingercount.FingerCountProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.objectdetection.ObjectDetectionProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.scenedescription.SceneDescriptionProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.florence.FlorenceProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama.PanoramaProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.sam.SamProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.videosegmentation.VideoSegmentationProcessor
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.vizlens.VizLensProcessor

/**
 * Singleton managing the lifecycle of all on-device processors.
 */
object OnDeviceProcessorManager {
    private const val TAG = "OnDeviceProcessorMgr"

    private val processors = mutableMapOf<Int, OnDeviceProcessor>()
    private var isInitialized = false
    private var samProcessor: SamProcessor? = null

    fun initialize(context: Context) {
        if (isInitialized) return

        val basicProcessor = BasicProcessor()
        val fingerCountProcessor = FingerCountProcessor()
        val objectDetectionProcessor = ObjectDetectionProcessor()
        val sceneDescriptionProcessor = SceneDescriptionProcessor()
        val vizLensProcessor = VizLensProcessor()
        val videoSegmentationProcessor = VideoSegmentationProcessor()
        val panoramaProcessor = PanoramaProcessor()
        val florenceProcessor = FlorenceProcessor()
        val samProc           = SamProcessor().also { samProcessor = it }

        processors[basicProcessor.id] = basicProcessor
        processors[fingerCountProcessor.id] = fingerCountProcessor
        processors[objectDetectionProcessor.id] = objectDetectionProcessor
        processors[sceneDescriptionProcessor.id] = sceneDescriptionProcessor
        processors[vizLensProcessor.id] = vizLensProcessor
        processors[videoSegmentationProcessor.id] = videoSegmentationProcessor
        processors[panoramaProcessor.id] = panoramaProcessor
        processors[florenceProcessor.id] = florenceProcessor

        // Give PanoramaProcessor references to Florence (scene analysis) and
        // SAM (mask segmentation) so it can enrich hierarchy nodes after stitching.
        panoramaProcessor.setFlorenceProcessor(florenceProcessor)
        panoramaProcessor.setSamProcessor(samProc)

        // SAM is not an OnDeviceProcessor (no frame pipeline) — initialize separately.
        samProc.initialize(context)

        for (processor in processors.values) {
            try {
                processor.initialize(context)
                Log.d(TAG, "Initialized processor: ${processor.name}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize ${processor.name}: ${e.message}")
            }
        }

        // Note: FastVLM engine will be loaded lazily when processor is used
        // to avoid memory issues at app launch

        isInitialized = true
        Log.d(TAG, "OnDeviceProcessorManager initialized with ${processors.size} processors")
    }

    fun getProcessor(id: Int): OnDeviceProcessor? = processors[id]

    fun getOnDeviceProcessorInfoList(): List<ProcessorInfo> {
        return processors.values.map { processor ->
            ProcessorInfo(
                id = processor.id,
                name = processor.name,
                description = processor.description,
                isOnDevice = true
            )
        }
    }

    fun isOnDeviceProcessor(id: Int): Boolean = processors.containsKey(id)

    fun release() {
        for (processor in processors.values) {
            try {
                processor.release()
            } catch (e: Exception) {
                Log.e(TAG, "Error releasing ${processor.name}: ${e.message}")
            }
        }
        processors.clear()
        samProcessor?.release()
        samProcessor  = null
        isInitialized = false
        Log.d(TAG, "OnDeviceProcessorManager released")
    }

    /** Exposed so callers (e.g. StreamViewModel) can access SAM directly for live RP use. */
    fun getSamProcessor(): SamProcessor? = samProcessor
}
