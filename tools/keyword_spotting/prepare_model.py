#!/usr/bin/env python3
"""
Train a 3-class keyword-spotting CNN on Google Speech Commands v2
and export it to ONNX for use with VoiceResponseListener.

Classes: yes=0, no=1, other=2

Usage (run once on your dev machine, ~5-10 min on CPU):
    pip install torch torchaudio
    python tools/keyword_spotting/prepare_model.py

The resulting ONNX is written to:
    selfdrive/modeld/models/keyword_spotting.onnx

Feature extraction exactly matches VoiceResponseListener._log_mel_spectrogram():
    16 kHz mono, STFT frame=400/hop=160/fft=512, 40 mel bins → [98, 40] per clip

Noise augmentation:
    Each yes/no utterance is duplicated 3× with additive noise at SNR 5-25 dB,
    covering the highway car-cabin range (~10-20 dB SNR for conversational speech).
    This makes the model robust to road noise, engine rumble, and HVAC.
"""

import sys
import time
import random
from pathlib import Path

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT  = Path(__file__).parent.parent.parent
MODEL_OUT  = REPO_ROOT / "selfdrive" / "modeld" / "models" / "keyword_spotting.onnx"
DATA_DIR   = REPO_ROOT / "tools" / "keyword_spotting" / "data"

# ── feature extraction (must match VoiceResponseListener) ─────────────────────

SAMPLE_RATE    = 16_000
WINDOW_SAMPLES = 16_000
_FFT_SIZE      = 512
_WIN_LENGTH    = 400
_HOP_LENGTH    = 160
_N_MELS        = 40
_F_MIN         = 20.0
_F_MAX         = 8_000.0


def _hz_to_mel(hz):
  return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
  return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank():
  mel_min = _hz_to_mel(_F_MIN)
  mel_max = _hz_to_mel(_F_MAX)
  mel_pts = np.linspace(mel_min, mel_max, _N_MELS + 2)
  hz_pts  = np.array([_mel_to_hz(m) for m in mel_pts])
  bins    = np.floor((_FFT_SIZE + 1) * hz_pts / SAMPLE_RATE).astype(int)
  n_freqs = _FFT_SIZE // 2 + 1
  fb      = np.zeros((n_freqs, _N_MELS), dtype=np.float32)
  for m in range(1, _N_MELS + 1):
    lo, ctr, hi = bins[m - 1], bins[m], bins[m + 1]
    if ctr > lo:
      fb[lo:ctr, m - 1] = (np.arange(lo, ctr) - lo) / (ctr - lo)
    if hi > ctr:
      fb[ctr:hi, m - 1] = (hi - np.arange(ctr, hi)) / (hi - ctr)
  return fb


_MEL_FB   = _build_mel_filterbank()
_HANN_WIN = np.hanning(_WIN_LENGTH).astype(np.float32)


def augment_with_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
  """Mix speech with additive white noise at the given SNR (dB).

  SNR guide for car cabins:
    ~20 dB  quiet city driving
    ~15 dB  normal highway speed
    ~10 dB  highway + windows down / loud HVAC
     ~5 dB  very loud environment (construction, etc.)
  """
  signal_power = np.mean(audio ** 2) + 1e-9
  noise_power  = signal_power / (10 ** (snr_db / 10))
  noise        = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
  return np.clip(audio + noise, -1.0, 1.0)


def extract_features(audio: np.ndarray) -> np.ndarray:
  """Convert raw float32 audio → log-mel spectrogram [98, 40]."""
  # Pad or trim to exactly WINDOW_SAMPLES
  if len(audio) < WINDOW_SAMPLES:
    audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
  else:
    audio = audio[:WINDOW_SAMPLES]

  n_frames = (WINDOW_SAMPLES - _WIN_LENGTH) // _HOP_LENGTH + 1
  frames   = np.lib.stride_tricks.sliding_window_view(audio, _WIN_LENGTH)[::_HOP_LENGTH][:n_frames]
  frames   = frames * _HANN_WIN
  spectra  = np.fft.rfft(frames, n=_FFT_SIZE)
  power    = (np.abs(spectra) ** 2).astype(np.float32)
  mel      = power @ _MEL_FB
  log_mel  = np.log(mel + 1e-6)
  return log_mel  # (98, 40)


# ── dataset ────────────────────────────────────────────────────────────────────

def load_dataset():
  """Download Google Speech Commands v2 via torchaudio and return (X, y) arrays."""
  try:
    import torch
    import torchaudio
  except ImportError:
    print("ERROR: install dependencies first:  pip install torch torchaudio")
    sys.exit(1)

  print("Downloading Speech Commands v2 (this may take a few minutes)...")
  DATA_DIR.mkdir(parents=True, exist_ok=True)

  TARGET_WORDS = {"yes", "no"}
  # torchaudio SPEECHCOMMANDS uses 'validation' and 'testing' splits
  X, y = [], []

  # SNR levels that cover the car-cabin range (highway ≈ 10-20 dB)
  NOISE_SNRS = [20.0, 12.0, 6.0]   # 3 augmented copies per utterance

  for split in ("training", "validation"):
    ds = torchaudio.datasets.SPEECHCOMMANDS(
      str(DATA_DIR), url="speech_commands_v0.02", download=True,
      subset=split,
    )
    for waveform, sample_rate, label, *_ in ds:
      if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
      audio = waveform[0].numpy()   # mono

      if label == "yes":
        cls = 0
      elif label == "no":
        cls = 1
      else:
        # Subsample "other" to keep the dataset balanced
        if random.random() < 0.15:
          X.append(extract_features(audio)); y.append(2)
        continue

      # Original clean version
      X.append(extract_features(audio)); y.append(cls)
      # Augmented noisy versions — teaches the model to handle car cabin audio
      for snr in NOISE_SNRS:
        X.append(extract_features(augment_with_noise(audio, snr))); y.append(cls)

  X = np.array(X, dtype=np.float32)   # (N, 98, 40)
  y = np.array(y, dtype=np.int64)
  counts = {c: int((y == c).sum()) for c in range(3)}
  print(f"Dataset (with noise augmentation): yes={counts[0]}  no={counts[1]}  other={counts[2]}")
  return X, y


# ── model ──────────────────────────────────────────────────────────────────────

def build_model():
  import torch.nn as nn

  class KeywordCNN(nn.Module):
    def __init__(self):
      super().__init__()
      self.features = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2, 2),                                         # → (32, 49, 20)
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2, 2),                                         # → (64, 24, 10)
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2, 2),                                         # → (128, 12, 5)
      )
      self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 12 * 5, 256), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(256, 3),
      )

    def forward(self, x):             # x: [B, 98, 40]
      x = x.unsqueeze(1)              # → [B, 1, 98, 40]
      return self.classifier(self.features(x))

  return KeywordCNN()


# ── training ───────────────────────────────────────────────────────────────────

def train(X, y):
  import torch
  import torch.nn as nn
  from torch.utils.data import TensorDataset, DataLoader

  device = torch.device("cpu")
  model  = build_model().to(device)

  # Normalise features
  mean, std = X.mean(), X.std()
  X = (X - mean) / (std + 1e-6)
  print(f"Feature mean={mean:.3f}  std={std:.3f}  (saved for normalisation)")

  # Train/val split
  n     = len(X)
  idx   = np.random.permutation(n)
  split = int(0.85 * n)
  X_tr, y_tr = X[idx[:split]], y[idx[:split]]
  X_va, y_va = X[idx[split:]], y[idx[split:]]

  ds_tr  = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
  ds_va  = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
  dl_tr  = DataLoader(ds_tr, batch_size=64, shuffle=True)
  dl_va  = DataLoader(ds_va, batch_size=256)

  opt       = torch.optim.Adam(model.parameters(), lr=1e-3)
  scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
  criterion = nn.CrossEntropyLoss()

  EPOCHS = 15
  best_acc = 0.0
  best_state = None

  for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in dl_tr:
      opt.zero_grad()
      loss = criterion(model(xb.to(device)), yb.to(device))
      loss.backward()
      opt.step()
      total_loss += loss.item() * len(xb)
    scheduler.step()

    model.eval()
    correct = 0
    with torch.no_grad():
      for xb, yb in dl_va:
        preds = model(xb.to(device)).argmax(1)
        correct += (preds == yb.to(device)).sum().item()
    acc = correct / len(ds_va)
    print(f"Epoch {epoch:2d}/{EPOCHS}  loss={total_loss/len(ds_tr):.4f}  val_acc={acc:.3f}")
    if acc > best_acc:
      best_acc = acc
      best_state = {k: v.clone() for k, v in model.state_dict().items()}

  print(f"\nBest validation accuracy: {best_acc:.3f}")
  model.load_state_dict(best_state)
  return model, mean, std


# ── export ─────────────────────────────────────────────────────────────────────

def export_onnx(model, feature_mean: float, feature_std: float):
  import torch

  model.eval()
  dummy = torch.zeros(1, 98, 40)

  MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    model,
    dummy,
    str(MODEL_OUT),
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
  )
  print(f"\nONNX model written to: {MODEL_OUT}")

  # Save normalisation stats alongside the model so VoiceResponseListener can
  # apply the same z-score normalisation at inference time.
  npz_out = MODEL_OUT.with_suffix(".npz")
  np.savez(str(npz_out), mean=np.float32(feature_mean), std=np.float32(feature_std))
  print(f"Normalisation stats written to: {npz_out}")


def verify_onnx():
  import onnxruntime as ort
  sess  = ort.InferenceSession(str(MODEL_OUT), providers=["CPUExecutionProvider"])
  dummy = np.zeros((1, 98, 40), dtype=np.float32)
  out   = sess.run(None, {"input": dummy})[0]
  assert out.shape == (1, 3), f"Unexpected output shape: {out.shape}"
  print(f"ONNX verification passed — output shape {out.shape}, classes: yes=0 no=1 other=2")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
  random.seed(42)
  np.random.seed(42)

  t0 = time.time()
  X, y = load_dataset()
  model, feat_mean, feat_std = train(X, y)
  export_onnx(model, feat_mean, feat_std)
  verify_onnx()
  print(f"\nDone in {time.time() - t0:.0f}s")
  print("\nNext steps:")
  print("  Local test:")
  print("    tools/replay/replay --demo   # terminal 1")
  print("    python selfdrive/ui/ui.py    # terminal 2")
  print("    python system/micd.py        # terminal 3")
  print("    touch /tmp/hazard_trigger    # trigger popup, then say yes/no")
  print()
  print("  Deploy to Comma 4:")
  print("    bash tools/keyword_spotting/deploy_model.sh <device-ip>")


if __name__ == "__main__":
  main()
