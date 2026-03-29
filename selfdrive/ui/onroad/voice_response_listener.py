"""
Voice keyword listener for hazard pop-up yes/no responses.

Subscribes to rawAudioData from micd, buffers 1-second sliding windows,
computes log-mel spectrograms with numpy, and runs a pre-trained ONNX
keyword-spotting model to detect "yes" or "no".

Expected model
--------------
  File:   selfdrive/modeld/models/keyword_spotting.onnx
  Input:  float32 [1, 98, 40]   (98 time frames × 40 mel bins, 1-second window)
  Output: float32 [1, 3]         (logits; classes: yes=0, no=1, other=2)

  Generate the model once on your dev machine:
      pip install torch torchaudio
      python tools/keyword_spotting/prepare_model.py
  The script downloads Google Speech Commands v2, trains a small CNN (~5 min),
  and writes the ONNX file to selfdrive/modeld/models/keyword_spotting.onnx.

Audio pipeline
--------------
  micd publishes int16 rawAudioData at 16 kHz, 50 ms chunks (800 samples).
  This listener subscribes on a daemon thread, fills a 16 000-sample ring
  buffer, and slides an inference window on every new chunk.  A queued
  result is consumed by HazardPopup._render() on the UI thread.
"""

import threading
import collections
import numpy as np
from pathlib import Path

import cereal.messaging as messaging

# ── model location ────────────────────────────────────────────────────────────

_MODEL_PATH = Path(__file__).parent.parent.parent / "modeld" / "models" / "keyword_spotting.onnx"

# Class indices for the 3-class model produced by tools/keyword_spotting/prepare_model.py
# Output logits order: [yes, no, other]
YES_IDX = 0
NO_IDX  = 1

# ── audio / feature constants ─────────────────────────────────────────────────

SAMPLE_RATE    = 16_000   # Hz (must match micd SAMPLE_RATE)
WINDOW_SAMPLES = 16_000   # 1-second inference window
HOP_SAMPLES    = 800      # 50 ms  (matches micd SAMPLE_BUFFER)

# STFT parameters — must match tools/keyword_spotting/prepare_model.py
_FFT_SIZE   = 512   # next power-of-2 above win_length
_WIN_LENGTH = 400   # 25 ms
_HOP_LENGTH = 160   # 10 ms  → (16000 - 400) // 160 + 1 = 98 frames
_N_MELS     = 40
_F_MIN      = 20.0
_F_MAX      = 8_000.0

CONFIDENCE_THRESHOLD = 0.70   # 0.70 handles real car audio (road/engine noise lowers model confidence)
NOISE_GATE_DB        = 85.0   # skip inference only in extreme noise; highway cabin is ~65-75 dB


# ── mel filterbank (computed once at import time) ─────────────────────────────

def _hz_to_mel(hz: float) -> float:
  return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
  return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(n_fft: int, n_mels: int, sr: int,
                          f_min: float, f_max: float) -> np.ndarray:
  """Return float32 filterbank matrix of shape (n_fft // 2 + 1, n_mels)."""
  mel_min = _hz_to_mel(f_min)
  mel_max = _hz_to_mel(f_max)
  mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
  hz_points = np.array([_mel_to_hz(m) for m in mel_points])
  # Map Hz to FFT bin indices
  bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
  n_freqs = n_fft // 2 + 1
  fb = np.zeros((n_freqs, n_mels), dtype=np.float32)
  for m in range(1, n_mels + 1):
    lo, ctr, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
    if ctr > lo:
      fb[lo:ctr, m - 1] = (np.arange(lo, ctr) - lo) / (ctr - lo)
    if hi > ctr:
      fb[ctr:hi, m - 1] = (hi - np.arange(ctr, hi)) / (hi - ctr)
  return fb


_MEL_FB = _build_mel_filterbank(_FFT_SIZE, _N_MELS, SAMPLE_RATE, _F_MIN, _F_MAX)
_HANN_WIN = np.hanning(_WIN_LENGTH).astype(np.float32)


# ── log-mel spectrogram ───────────────────────────────────────────────────────

def _log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
  """
  Compute a log-mel spectrogram matching the TF tutorial preprocessing.

  Args:
    audio: float32 array of shape (WINDOW_SAMPLES,), normalised to [-1, 1]

  Returns:
    float32 array of shape (98, 40)  [time_frames × mel_bins]
  """
  n_frames = (len(audio) - _WIN_LENGTH) // _HOP_LENGTH + 1
  frames = np.lib.stride_tricks.sliding_window_view(audio, _WIN_LENGTH)[::_HOP_LENGTH][:n_frames]
  frames = frames * _HANN_WIN                      # apply window
  spectra = np.fft.rfft(frames, n=_FFT_SIZE)       # (n_frames, _FFT_SIZE//2+1)
  power   = (np.abs(spectra) ** 2).astype(np.float32)
  mel     = power @ _MEL_FB                         # (n_frames, _N_MELS)
  log_mel = np.log(mel + 1e-6)
  return log_mel                                     # (98, 40)


# ── onnxruntime session (loaded lazily) ───────────────────────────────────────

_session_lock  = threading.Lock()
_ort_session   = None
_feat_mean: float = 0.0
_feat_std:  float = 1.0


def _get_session():
  global _ort_session, _feat_mean, _feat_std
  if _ort_session is not None:
    return _ort_session
  with _session_lock:
    if _ort_session is not None:
      return _ort_session
    try:
      import onnxruntime as ort  # type: ignore[import]
    except ImportError:
      return None
    if not _MODEL_PATH.exists():
      return None
    # Load normalisation stats saved by prepare_model.py alongside the ONNX
    npz_path = _MODEL_PATH.with_suffix(".npz")
    if npz_path.exists():
      stats = np.load(str(npz_path))
      _feat_mean = float(stats["mean"])
      _feat_std  = float(stats["std"])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    _ort_session = ort.InferenceSession(str(_MODEL_PATH), sess_options=opts,
                                        providers=["CPUExecutionProvider"])
    # Warmup: run one dummy inference so the first real call isn't slow
    dummy = np.zeros((1, 98, 40), dtype=np.float32)
    input_name = _ort_session.get_inputs()[0].name
    _ort_session.run(None, {input_name: dummy})
    return _ort_session


# ── main class ────────────────────────────────────────────────────────────────

class VoiceResponseListener:
  """
  Listens for "yes" or "no" in the microphone stream while active.

  Usage:
      listener = VoiceResponseListener()
      listener.start()                   # when popup appears
      listener.stop()                    # when popup is dismissed
      answer = listener.consume_pending()  # call each render frame; returns "yes"/"no"/None
  """

  def __init__(self) -> None:
    self._thread: threading.Thread | None = None
    self._stop_event = threading.Event()
    self._pending: str | None = None          # set from bg thread, read on UI thread
    self._pending_lock = threading.Lock()
    self._ambient_db: float = 0.0             # updated from soundPressure messages

  # ── public API ───────────────────────────────────────────────────────────────

  def start(self) -> None:
    """Start background listening thread. Silent no-op if model is absent."""
    if self._thread and self._thread.is_alive():
      return
    if _get_session() is None:
      return   # model unavailable — silently degrade, touch buttons still work
    self._stop_event.clear()
    with self._pending_lock:
      self._pending = None
    self._thread = threading.Thread(target=self._listen_loop, daemon=True,
                                    name="voice_response_listener")
    self._thread.start()

  def stop(self) -> None:
    """Signal the background thread to exit. Non-blocking."""
    self._stop_event.set()

  def consume_pending(self) -> str | None:
    """
    Return and clear any pending answer detected by the background thread.
    Must be called from the render (UI) thread only.
    """
    with self._pending_lock:
      answer = self._pending
      self._pending = None
    return answer

  # ── background thread ────────────────────────────────────────────────────────

  def _listen_loop(self) -> None:
    sm = messaging.SubMaster(['rawAudioData', 'soundPressure'])
    ring: collections.deque[np.ndarray] = collections.deque(
      maxlen=WINDOW_SAMPLES // HOP_SAMPLES
    )
    last_inference = 0.0
    INFERENCE_INTERVAL = 0.25  # run inference at most every 250ms

    while not self._stop_event.is_set():
      sm.update(timeout=100)   # 100 ms poll timeout

      if sm.updated['soundPressure']:
        self._ambient_db = sm['soundPressure'].soundPressureWeightedDb

      if not sm.updated['rawAudioData']:
        continue

      raw = sm['rawAudioData']
      chunk = np.frombuffer(raw.data, dtype=np.int16).astype(np.float32) / 32768.0
      ring.append(chunk)

      # Need a full 1-second window before running inference
      if len(ring) < ring.maxlen:
        continue

      # Noise gate: skip inference in very loud conditions
      if self._ambient_db > NOISE_GATE_DB:
        continue

      # Throttle inference to avoid overwhelming the CPU
      now = time.monotonic()
      if now - last_inference < INFERENCE_INTERVAL:
        continue
      last_inference = now

      audio = np.concatenate(list(ring))
      self._run_inference(audio)

  def _run_inference(self, audio: np.ndarray) -> None:
    session = _get_session()
    if session is None:
      return

    log_mel = _log_mel_spectrogram(audio)                        # (98, 40)
    log_mel = (log_mel - _feat_mean) / (_feat_std + 1e-6)       # normalise
    inp = log_mel[np.newaxis, :, :]                              # (1, 98, 40) float32

    try:
      input_name = session.get_inputs()[0].name
      logits = session.run(None, {input_name: inp})[0]   # (1, N_CLASSES)
    except Exception:
      return

    probs = _softmax(logits[0])
    yes_conf = float(probs[YES_IDX])
    no_conf  = float(probs[NO_IDX])

    if yes_conf >= CONFIDENCE_THRESHOLD:
      self._queue_answer("yes")
    elif no_conf >= CONFIDENCE_THRESHOLD:
      self._queue_answer("no")

  def _queue_answer(self, answer: str) -> None:
    with self._pending_lock:
      if self._pending is None:   # first detection wins
        self._pending = answer
    self._stop_event.set()        # stop listening after first confident answer


# ── helpers ───────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
  x = x - x.max()
  e = np.exp(x)
  return e / e.sum()
