/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Explore
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Timer
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.meta.wearable.dat.externalsampleapps.cameraaccess.R
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.CaptureButtonMode
import com.meta.wearable.dat.externalsampleapps.cameraaccess.stream.TimerMode

@Composable
fun CircleButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    content: @Composable RowScope.() -> Unit,
) {
  Button(
      modifier = modifier.aspectRatio(1f),
      onClick = onClick,
      colors = ButtonDefaults.buttonColors(containerColor = Color.White),
      shape = CircleShape,
      contentPadding = PaddingValues(0.dp),
      content = content,
  )
}

@Composable
fun TimerButton(
    timerMode: TimerMode,
    onClick: () -> Unit,
) {
  CircleButton(onClick = onClick) {
    if (timerMode == TimerMode.UNLIMITED) {
      Icon(
          imageVector = Icons.Filled.Timer,
          contentDescription = stringResource(timerMode.labelId),
          tint = Color.Black,
      )
    } else {
      Text(
          text = stringResource(timerMode.labelId),
          color = Color.Black,
      )
    }
  }
}

@Composable
fun CaptureButton(
    mode: CaptureButtonMode,
    onClick: () -> Unit,
) {
    val (icon, containerColor, contentColor) = when (mode) {
        CaptureButtonMode.CAMERA ->
            Triple(Icons.Filled.PhotoCamera, Color.White, Color.Black)
        CaptureButtonMode.RECORDING ->
            Triple(Icons.Filled.Stop, AppColor.Red, Color.White)
        CaptureButtonMode.PANORAMA_DONE ->
            Triple(Icons.Filled.Explore, AppColor.DeepBlue, Color.White)
        CaptureButtonMode.PROXY_ACTIVE ->
            Triple(Icons.Filled.Close, AppColor.Yellow, Color.Black)
    }
    Button(
        modifier = Modifier.aspectRatio(1f),
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(containerColor = containerColor),
        shape = CircleShape,
        contentPadding = PaddingValues(0.dp),
    ) {
        Icon(
            imageVector = icon,
            contentDescription = stringResource(R.string.capture_photo),
            tint = contentColor,
        )
    }
}
