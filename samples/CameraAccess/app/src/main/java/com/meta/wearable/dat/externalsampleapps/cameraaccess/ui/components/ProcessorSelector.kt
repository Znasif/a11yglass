/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.meta.wearable.dat.externalsampleapps.cameraaccess.network.models.ProcessorInfo
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.AppColor
import kotlinx.collections.immutable.ImmutableList

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ProcessorSelector(
    processors: ImmutableList<ProcessorInfo>,
    selectedProcessorId: Int,
    onProcessorSelected: (Int) -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    var expanded by remember { mutableStateOf(false) }
    val selectedProcessor = processors.find { it.id == selectedProcessorId }
    
    Column(modifier = modifier.fillMaxWidth()) {
        Text(
            text = "Processor Mode",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        
        Spacer(modifier = Modifier.height(4.dp))
        
        ExposedDropdownMenuBox(
            expanded = expanded,
            onExpandedChange = { if (enabled) expanded = it }
        ) {
            OutlinedTextField(
                value = selectedProcessor?.name ?: "Select Processor",
                onValueChange = {},
                readOnly = true,
                enabled = enabled,
                trailingIcon = {
                    ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
                },
                modifier = Modifier
                    .menuAnchor()
                    .fillMaxWidth(),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = AppColor.DeepBlue,
                    unfocusedBorderColor = MaterialTheme.colorScheme.outline
                )
            )
            
            ExposedDropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                if (processors.isEmpty()) {
                    DropdownMenuItem(
                        text = { 
                            Text(
                                "No processors available",
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            ) 
                        },
                        onClick = { expanded = false },
                        enabled = false
                    )
                } else {
                    processors.forEach { processor ->
                        DropdownMenuItem(
                            text = {
                                Column {
                                    Text(
                                        text = processor.name,
                                        style = MaterialTheme.typography.bodyLarge
                                    )
                                    if (processor.description.isNotBlank()) {
                                        Text(
                                            text = processor.description,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                    }
                                }
                            },
                            onClick = {
                                onProcessorSelected(processor.id)
                                expanded = false
                            },
                            leadingIcon = {
                                RadioButton(
                                    selected = processor.id == selectedProcessorId,
                                    onClick = null
                                )
                            }
                        )
                    }
                }
            }
        }
        
        // Show selected processor description
        selectedProcessor?.description?.takeIf { it.isNotBlank() }?.let { description ->
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}
