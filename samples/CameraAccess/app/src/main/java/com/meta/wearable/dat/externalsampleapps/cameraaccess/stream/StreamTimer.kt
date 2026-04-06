/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// StreamTimer - Auto-Stop Timer for DAT Streaming Sessions
//
// This utility class provides automatic timeout functionality.

package com.meta.wearable.dat.externalsampleapps.cameraaccess.stream

import android.os.CountDownTimer
import com.meta.wearable.dat.externalsampleapps.cameraaccess.R
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class TimerMode(val durationSeconds: Long, val labelId: Int) {
  UNLIMITED(0L, R.string.unlimited_time),
  ONE_MINUTE(60L, R.string.one_minute),
  FIVE_MINUTE(300L, R.string.five_minute),
  TEN_MINUTE(600L, R.string.ten_minute),
  FIFTEEN_MINUTE(900L, R.string.fifteen_minute);

  fun next(): TimerMode {
    return when (this) {
      UNLIMITED -> ONE_MINUTE
      ONE_MINUTE -> FIVE_MINUTE
      FIVE_MINUTE -> TEN_MINUTE
      TEN_MINUTE -> FIFTEEN_MINUTE
      FIFTEEN_MINUTE -> UNLIMITED
    }
  }
}

class StreamTimer {
  private var countDownTimer: CountDownTimer? = null

  private val _timerMode = MutableStateFlow(TimerMode.UNLIMITED)
  val timerMode: StateFlow<TimerMode> = _timerMode.asStateFlow()

  private val _remainingTimeSeconds = MutableStateFlow<Long?>(null)
  val remainingTimeSeconds: StateFlow<Long?> = _remainingTimeSeconds.asStateFlow()

  private val _isTimerExpired = MutableStateFlow(false)
  val isTimerExpired: StateFlow<Boolean> = _isTimerExpired.asStateFlow()

  fun cycleTimerMode() {
    val newMode = _timerMode.value.next()
    _timerMode.value = newMode

    // Reset timer state when mode changes
    stopTimer()
    _remainingTimeSeconds.value =
        if (newMode == TimerMode.UNLIMITED) null else newMode.durationSeconds
    _isTimerExpired.value = false
  }

  fun startTimer() {
    val currentMode = _timerMode.value

    if (currentMode == TimerMode.UNLIMITED) {
      _remainingTimeSeconds.value = null
      return
    }

    stopTimer() // Stop any existing timer

    val durationMs = currentMode.durationSeconds * 1000L
    _remainingTimeSeconds.value = currentMode.durationSeconds
    _isTimerExpired.value = false

    countDownTimer =
        object : CountDownTimer(durationMs, 1000L) {
              override fun onTick(millisUntilFinished: Long) {
                _remainingTimeSeconds.value = millisUntilFinished / 1000L
              }

              override fun onFinish() {
                _remainingTimeSeconds.value = 0L
                _isTimerExpired.value = true
              }
            }
            .start()
  }

  fun stopTimer() {
    countDownTimer?.cancel()
    countDownTimer = null
  }

  fun resetTimer() {
    stopTimer()
    _timerMode.value = TimerMode.UNLIMITED
    _remainingTimeSeconds.value = null
    _isTimerExpired.value = false
  }

  fun cleanup() {
    stopTimer()
  }
}
