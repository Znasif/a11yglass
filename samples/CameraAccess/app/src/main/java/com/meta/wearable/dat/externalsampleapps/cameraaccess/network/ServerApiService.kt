/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.network

import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ProcessorsResponse
import retrofit2.Response
import retrofit2.http.GET

/**
 * Retrofit service interface for server REST API calls.
 */
interface ServerApiService {
    
    /**
     * Fetch available processors from the server.
     * Endpoint: GET /processors
     */
    @GET("processors")
    suspend fun getProcessors(): Response<ProcessorsResponse>
}
