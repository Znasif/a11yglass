/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.panorama

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.BiasAlignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama.NODE_COLORS
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama.POINTING_FRACTION
import com.meta.wearable.dat.externalsampleapps.cameraaccess.processor.panorama.SavedPanorama
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.CaptureButtonMode
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.StreamUiState
import com.meta.wearable.dat.externalsampleapps.cameraaccess.ui.CaptureButton
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Panorama viewer overlay. Renders inside the main full-screen Box in StreamScreen.
 *
 * Handles:
 * - Panorama image display with FillHeight + BiasAlignment pan
 * - Horizontal swipe: discrete node stepping (PANORAMA_DONE with nodes)
 *                     or continuous pan (PROXY_ACTIVE / no nodes)
 * - Node overlay buttons with focus highlight and TalkBack semantics
 * - Live camera cutout in Reality Proxy mode
 * - Floating exit button in Reality Proxy mode
 *
 * Only renders when [streamUiState.carouselPanorama] is non-null.
 */
@Composable
fun BoxScope.PanoramaOverlay(
    streamUiState: StreamUiState,
    onStepNode: (Int) -> Unit,
    onCapturePhoto: () -> Unit,
) {
    val panorama = streamUiState.carouselPanorama ?: return

    // Manual pan position — reset to centre when a new panorama is loaded.
    var manualXFraction by remember { mutableStateOf(0.5f) }
    LaunchedEffect(panorama) { manualXFraction = 0.5f }

    // Snap pan to focused node when currentNodeIndex changes.
    val currentNodeIndex = streamUiState.currentNodeIndex
    LaunchedEffect(currentNodeIndex) {
        val nodes = streamUiState.hierarchyNodes
        if (currentNodeIndex in nodes.indices) {
            manualXFraction = nodes[currentNodeIndex].panoramaXFraction
        }
    }

    val xFraction = if (streamUiState.captureButtonMode == CaptureButtonMode.PROXY_ACTIVE)
        streamUiState.carouselXFraction else manualXFraction

    // Discrete stepping when nodes are present and not in proxy mode;
    // continuous pan otherwise.
    val discreteSwipe = streamUiState.captureButtonMode == CaptureButtonMode.PANORAMA_DONE
        && streamUiState.hierarchyNodes.isNotEmpty()

    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        // ── Panorama image ──────────────────────────────────────────────────
        Image(
            bitmap = panorama.asImageBitmap(),
            contentDescription = null,
            modifier = Modifier
                .fillMaxSize()
                .pointerInput(discreteSwipe) {
                    if (discreteSwipe) {
                        var accumulated = 0f
                        detectHorizontalDragGestures(
                            onDragEnd = { accumulated = 0f },
                            onDragCancel = { accumulated = 0f },
                        ) { _, dragAmount ->
                            accumulated += dragAmount
                            val threshold = size.width * 0.15f
                            if (accumulated > threshold) {
                                onStepNode(-1); accumulated = 0f
                            } else if (accumulated < -threshold) {
                                onStepNode(+1); accumulated = 0f
                            }
                        }
                    } else {
                        detectHorizontalDragGestures { _, dragAmount ->
                            manualXFraction = (manualXFraction - dragAmount / size.width * 2f)
                                .coerceIn(0f, 1f)
                        }
                    }
                },
            contentScale = ContentScale.FillHeight,
            alignment = BiasAlignment(horizontalBias = 2f * xFraction - 1f, verticalBias = 0f),
        )

        // ── Coordinate mapping ──────────────────────────────────────────────
        // displayedW: width of the panorama when its height fills the screen.
        // viewOffsetX: screen-space x of the panorama's left edge (negative when
        //              the panorama is wider than the screen).
        val displayedW: Dp = maxHeight * panorama.width / panorama.height
        val viewOffsetX: Dp = (maxWidth - displayedW) * xFraction

        // ── Node overlay buttons ────────────────────────────────────────────
        streamUiState.hierarchyNodes.forEachIndexed { idx, node ->
            val centerX = viewOffsetX + displayedW * node.panoramaXFraction
            val nodeW   = maxOf(displayedW * node.panoramaWidthFraction, 72.dp)
            val left    = centerX - nodeW / 2
            if (left > maxWidth || left + nodeW < 0.dp) return@forEachIndexed
            val centerY = maxHeight * node.panoramaYFraction
            val nodeH   = 36.dp
            val top     = (centerY - nodeH / 2).coerceIn(0.dp, maxHeight - nodeH)
            val color       = Color(NODE_COLORS[idx % NODE_COLORS.size])
            val isFocused   = idx == currentNodeIndex
            val borderColor = if (isFocused) Color.Yellow else Color.White.copy(alpha = 0.7f)
            val borderWidth = if (isFocused) 2.dp else 1.dp
            Box(
                modifier = Modifier
                    .offset(x = left, y = top)
                    .size(nodeW, nodeH)
                    .border(borderWidth, borderColor, RoundedCornerShape(8.dp))
                    .clip(RoundedCornerShape(8.dp))
                    .background(color.copy(alpha = if (isFocused) 0.95f else 0.75f))
                    .semantics(mergeDescendants = true) {
                        contentDescription = node.label
                        role = Role.Button
                    }
                    .clickable { manualXFraction = node.panoramaXFraction }
                    .padding(horizontal = 6.dp, vertical = 2.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = node.label,
                    color = Color.White,
                    fontSize = 10.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }

        // ── Reality Proxy: live camera cutout ───────────────────────────────
        if (streamUiState.captureButtonMode == CaptureButtonMode.PROXY_ACTIVE) {
            streamUiState.videoFrame?.let { live ->
                val cutoutW = maxWidth * POINTING_FRACTION
                val cutoutH = cutoutW * live.height / live.width
                Box(
                    modifier = Modifier
                        .size(cutoutW, cutoutH)
                        .align(Alignment.Center)
                        .border(2.dp, Color.White, RoundedCornerShape(4.dp))
                        .clip(RoundedCornerShape(4.dp)),
                ) {
                    Image(
                        bitmap = live.asImageBitmap(),
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.Crop,
                    )
                }
            }

            // Floating exit button (bottom centre — replaces hidden bottom controls bar)
            Box(
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .navigationBarsPadding()
                    .padding(bottom = 16.dp)
                    .size(56.dp),
            ) {
                CaptureButton(mode = CaptureButtonMode.PROXY_ACTIVE, onClick = onCapturePhoto)
            }
        }
    }
}

/** Gallery button shown in the secondary controls row when the panorama processor is active. */
@Composable
fun PanoramaGalleryButton(onClick: () -> Unit) {
    IconButton(
        onClick = onClick,
        modifier = Modifier
            .size(48.dp)
            .background(
                Color(0xFF1565C0), // AppColor.DeepBlue equivalent
                RoundedCornerShape(8.dp),
            ),
    ) {
        Icon(
            imageVector = Icons.Default.PhotoLibrary,
            contentDescription = "Saved panoramas",
            tint = Color.White,
        )
    }
}

/** Dialog listing saved panoramas with load and delete actions. */
@Composable
fun PanoramaPickerDialog(
    panoramas: List<SavedPanorama>,
    onSelect: (String) -> Unit,
    onDelete: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    val dateFormat = remember { SimpleDateFormat("MMM d, HH:mm", Locale.getDefault()) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Saved Panoramas") },
        text = {
            if (panoramas.isEmpty()) {
                Text(
                    text = "No saved panoramas yet.\nComplete a panorama sweep to save one automatically.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.Gray,
                )
            } else {
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(panoramas, key = { it.id }) { pano ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .background(Color.DarkGray.copy(alpha = 0.4f))
                                .clickable { onSelect(pano.id) }
                                .padding(8.dp),
                            horizontalArrangement = Arrangement.spacedBy(12.dp),
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            pano.thumbnailBitmap?.let { thumb ->
                                Image(
                                    bitmap = thumb.asImageBitmap(),
                                    contentDescription = null,
                                    modifier = Modifier
                                        .size(width = 96.dp, height = 54.dp)
                                        .clip(RoundedCornerShape(4.dp)),
                                    contentScale = ContentScale.Crop,
                                )
                            } ?: Box(
                                modifier = Modifier
                                    .size(width = 96.dp, height = 54.dp)
                                    .clip(RoundedCornerShape(4.dp))
                                    .background(Color.Gray.copy(alpha = 0.5f)),
                            )
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = dateFormat.format(Date(pano.timestamp)),
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = Color.White,
                                )
                                Text(
                                    text = "${pano.nodeCount} region${if (pano.nodeCount == 1) "" else "s"}",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = Color.Gray,
                                )
                            }
                            IconButton(
                                onClick = { onDelete(pano.id) },
                                modifier = Modifier.size(36.dp),
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Delete,
                                    contentDescription = "Delete",
                                    tint = Color.Red.copy(alpha = 0.8f),
                                )
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        },
    )
}
