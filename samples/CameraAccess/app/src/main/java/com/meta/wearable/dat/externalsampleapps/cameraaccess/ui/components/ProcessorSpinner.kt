/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ProcessorInfo
import kotlinx.collections.immutable.ImmutableList

/**
 * Compact spinner-style processor selector for use in StreamScreen.
 * Designed to overlay on video feed with semi-transparent background.
 */
@Composable
fun ProcessorSpinner(
    processors: ImmutableList<ProcessorInfo>,
    selectedProcessorId: Int,
    onProcessorSelected: (Int) -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    var expanded by remember { mutableStateOf(false) }
    val selectedProcessor = processors.find { it.id == selectedProcessorId }

    Box(modifier = modifier) {
        // Trigger button
        Surface(
            onClick = { if (enabled && processors.isNotEmpty()) expanded = true },
            modifier = Modifier
                .clip(RoundedCornerShape(8.dp)),
            color = Color.Black.copy(alpha = 0.6f),
            contentColor = Color.White
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = selectedProcessor?.name ?: "Select Processor",
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (enabled) Color.White else Color.Gray
                )
                Icon(
                    imageVector = Icons.Default.ArrowDropDown,
                    contentDescription = "Select processor",
                    tint = if (enabled) Color.White else Color.Gray,
                    modifier = Modifier.size(20.dp)
                )
            }
        }

        // Dropdown menu
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false },
            modifier = Modifier
                .background(Color(0xFF1A1A2E))
        ) {
            if (processors.isEmpty()) {
                DropdownMenuItem(
                    text = {
                        Text(
                            "No processors available",
                            color = Color.Gray
                        )
                    },
                    onClick = { expanded = false },
                    enabled = false
                )
            } else {
                processors.forEach { processor ->
                    DropdownMenuItem(
                        text = {
                            Text(
                                text = processor.name,
                                color = if (processor.id == selectedProcessorId) {
                                    Color.Cyan
                                } else {
                                    Color.White
                                }
                            )
                        },
                        onClick = {
                            onProcessorSelected(processor.id)
                            expanded = false
                        },
                        leadingIcon = if (processor.id == selectedProcessorId) {
                            {
                                Box(
                                    modifier = Modifier
                                        .size(8.dp)
                                        .background(Color.Cyan, RoundedCornerShape(4.dp))
                                )
                            }
                        } else null
                    )
                }
            }
        }
    }
}