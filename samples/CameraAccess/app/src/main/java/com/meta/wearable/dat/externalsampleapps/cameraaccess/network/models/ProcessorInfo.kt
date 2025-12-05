/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models

import com.google.gson.annotations.SerializedName

/**
 * Represents a processor available on the server.
 * Corresponds to the /processors endpoint response.
 */
data class ProcessorInfo(
    @SerializedName("id")
    val id: Int,
    
    @SerializedName("name")
    val name: String,
    
    @SerializedName("dependencies")
    val dependencies: List<Int> = emptyList(),
    
    @SerializedName("expects_input")
    val expectsInput: String = "image",
    
    @SerializedName("description")
    val description: String = ""
)

/**
 * Response wrapper for the /processors endpoint.
 */
data class ProcessorsResponse(
    @SerializedName("processors")
    val processors: List<ProcessorInfo>
)
