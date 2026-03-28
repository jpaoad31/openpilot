"""
Geofence-based fake bump triggers for demo/testing with recorded rides.

Drop a JSON file at /tmp/roadpass_demo_triggers.json:

  [
    {"lat": 37.7749, "lon": -122.4194, "radius_m": 30, "probability": 1.0, "label": "pothole"},
    {"lat": 37.7755, "lon": -122.4180, "radius_m": 25, "probability": 0.5, "label": "speed bump"}
  ]

Fields:
  lat, lon     — required, trigger center
  radius_m     — optional, default 30
  probability  — optional, 0.0–1.0, default 1.0 (always fires)
  label        — optional, for your reference only

Each trigger fires at most once per approach. The probability roll happens when
the device first enters the radius. The trigger rearms after the device leaves.
The file is re-read every 10 seconds so you can edit it live over SSH.
"""
import json
import os
import random
import time

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.onroad.hazard_fetcher import _haversine

TRIGGERS_PATH = "/tmp/roadpass_demo_triggers.json"
_RELOAD_INTERVAL = 10.0  # seconds


class DemoTriggers:
  def __init__(self):
    self._triggers: list[dict] = []
    self._last_load: float = 0.0
    # Track which triggers the device is currently inside, to fire only on entry.
    # Maps trigger index → True if inside radius on last check.
    self._inside: dict[int, bool] = {}
    # Triggers that fired and are cooling down until the device leaves.
    self._cooldown: set[int] = set()

  def check(self, lat: float, lon: float) -> bool:
    """
    Returns True on the frame a demo trigger fires.
    Call once per render frame with current GPS position.
    """
    self._maybe_reload()
    if not self._triggers:
      return False

    for idx, t in enumerate(self._triggers):
      dist = _haversine(lat, lon, t["lat"], t["lon"])
      radius = t.get("radius_m", 30)
      inside = dist <= radius

      was_inside = self._inside.get(idx, False)
      self._inside[idx] = inside

      if inside and not was_inside and idx not in self._cooldown:
        # Just entered the radius — roll probability
        prob = t.get("probability", 1.0)
        if random.random() < prob:
          self._cooldown.add(idx)
          label = t.get("label", f"trigger {idx}")
          cloudlog.info(f"DemoTriggers: fired [{label}] dist={dist:.0f}m prob={prob}")
          return True
        else:
          # Roll failed — cooldown so we don't re-roll every frame
          self._cooldown.add(idx)

      # Rearm once the device leaves the radius
      if not inside and idx in self._cooldown:
        self._cooldown.discard(idx)

    return False

  def _maybe_reload(self) -> None:
    now = time.monotonic()
    if now - self._last_load < _RELOAD_INTERVAL:
      return
    self._last_load = now

    if not os.path.exists(TRIGGERS_PATH):
      if self._triggers:
        cloudlog.info("DemoTriggers: file removed, clearing triggers")
      self._triggers = []
      return

    try:
      with open(TRIGGERS_PATH, "r") as f:
        data = json.load(f)
      if isinstance(data, list):
        self._triggers = data
        cloudlog.info(f"DemoTriggers: loaded {len(data)} trigger(s)")
      else:
        cloudlog.error("DemoTriggers: file must be a JSON array")
        self._triggers = []
    except Exception as e:
      cloudlog.error(f"DemoTriggers: failed to load: {e}")
