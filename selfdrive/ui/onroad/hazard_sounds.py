"""
Raylib audio for RoadPass hazard alerts.
Uses existing .wav assets in selfdrive/assets/sounds/.
Audio device is initialized lazily on first play; sounds are loaded once and cached.
"""
import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.common.swaglog import cloudlog

_SOUNDS_DIR = BASEDIR + "/selfdrive/assets/sounds/"
_audio_ready = False
_cache: dict[str, rl.Sound] = {}


def _ensure_audio() -> bool:
  global _audio_ready
  if _audio_ready:
    return True
  try:
    rl.init_audio_device()
    _audio_ready = True
  except Exception as e:
    cloudlog.error(f"hazard_sounds: failed to init audio: {e}")
  return _audio_ready


def _play(filename: str) -> None:
  if not _ensure_audio():
    return
  if filename not in _cache:
    _cache[filename] = rl.load_sound(_SOUNDS_DIR + filename)
  rl.play_sound(_cache[filename])


def play_hazard_detected() -> None:
  """Urgent alert when the bump confirmation popup appears."""
  _play("warning_immediate.wav")


def play_hazard_ahead() -> None:
  """Softer alert when an upcoming hazard card first appears."""
  _play("warning_soft.wav")
