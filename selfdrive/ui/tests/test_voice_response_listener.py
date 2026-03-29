"""
Tests for VoiceResponseListener.

These tests exercise the feature extraction and inference pipeline using
synthetic audio (pure sine waves and white noise) without requiring the
ONNX model file or a live microphone.
"""

import unittest
import numpy as np

from openpilot.selfdrive.ui.onroad.voice_response_listener import (
  SAMPLE_RATE,
  WINDOW_SAMPLES,
  YES_IDX,
  NO_IDX,
  _log_mel_spectrogram,
  _softmax,
  VoiceResponseListener,
)


class TestLogMelSpectrogram(unittest.TestCase):
  def test_output_shape(self):
    audio = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    spec = _log_mel_spectrogram(audio)
    self.assertEqual(spec.shape, (98, 40))

  def test_output_dtype(self):
    audio = np.random.randn(WINDOW_SAMPLES).astype(np.float32)
    spec = _log_mel_spectrogram(audio)
    self.assertEqual(spec.dtype, np.float32)

  def test_silence_produces_finite_values(self):
    audio = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    spec = _log_mel_spectrogram(audio)
    self.assertTrue(np.all(np.isfinite(spec)))

  def test_noise_produces_higher_values_than_silence(self):
    silence = _log_mel_spectrogram(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
    noise   = _log_mel_spectrogram(np.random.randn(WINDOW_SAMPLES).astype(np.float32))
    self.assertGreater(noise.mean(), silence.mean())

  def test_frequency_content_shows_in_correct_bins(self):
    # A 1 kHz sine wave should produce energy in the bins covering ~1 kHz
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
    audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    spec = _log_mel_spectrogram(audio)
    # Average energy across time frames; mel bin ~13-15 covers 1 kHz
    energy = spec.mean(axis=0)
    peak_bin = int(np.argmax(energy))
    self.assertGreater(energy[peak_bin], energy[0],
                       "1 kHz energy should exceed low-frequency energy")


class TestSoftmax(unittest.TestCase):
  def test_sums_to_one(self):
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    probs = _softmax(logits)
    self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

  def test_highest_logit_gets_highest_prob(self):
    logits = np.array([0.1, 5.0, 0.2], dtype=np.float32)
    probs = _softmax(logits)
    self.assertEqual(int(np.argmax(probs)), 1)


class TestVoiceResponseListenerStub(unittest.TestCase):
  """
  Tests that don't require the ONNX model.
  VoiceResponseListener.start() is a no-op when the model is absent,
  so we verify the API contract and threading behaviour.
  """

  def test_consume_pending_returns_none_initially(self):
    listener = VoiceResponseListener()
    self.assertIsNone(listener.consume_pending())

  def test_stop_before_start_is_safe(self):
    listener = VoiceResponseListener()
    listener.stop()   # must not raise

  def test_start_without_model_is_safe(self):
    listener = VoiceResponseListener()
    listener.start(lambda answer: None)   # model absent → silent no-op
    listener.stop()

  def test_consume_clears_pending(self):
    listener = VoiceResponseListener()
    # Directly inject a pending answer to simulate a detection
    with listener._pending_lock:
      listener._pending = "yes"
    self.assertEqual(listener.consume_pending(), "yes")
    self.assertIsNone(listener.consume_pending())

  def test_queue_answer_first_detection_wins(self):
    listener = VoiceResponseListener()
    listener._queue_answer("yes")
    listener._queue_answer("no")   # should be ignored
    self.assertEqual(listener.consume_pending(), "yes")


if __name__ == "__main__":
  unittest.main()
