"""
Geofence-based fake bump triggers for demo/testing with recorded rides.

Each trigger fires at most once per approach. The probability roll happens when
the device first enters the radius. The trigger rearms after the device leaves.
"""
import random

from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.onroad.hazard_fetcher import _haversine

DEMO_TRIGGERS = [
  {"lat": 32.83811, "lon": -117.23467, "radius_m": 5, "probability": 1.0, "label": "hazard near 32.838/-117.235"},
  {"lat": 32.79748, "lon": -117.20979, "radius_m": 5, "probability": 0.5, "label": "hazard near 32.797/-117.210"},
  {"lat": 32.75845, "lon": -117.20352, "radius_m": 5, "probability": 0.2, "label": "hazard near 32.758/-117.204"},
]

class DemoTriggers:
  def __init__(self):
    self._inside: dict[int, bool] = {}
    self._cooldown: set[int] = set()

  def check(self, lat: float, lon: float) -> bool:
    """
    Returns True on the frame a demo trigger fires.
    Call once per render frame with current GPS position.
    """
    for idx, t in enumerate(DEMO_TRIGGERS):
      dist = _haversine(lat, lon, t["lat"], t["lon"])
      inside = dist <= t.get("radius_m", 30)

      was_inside = self._inside.get(idx, False)
      self._inside[idx] = inside

      if inside and not was_inside and idx not in self._cooldown:
        prob = t.get("probability", 1.0)
        if random.random() < prob:
          self._cooldown.add(idx)
          label = t.get("label", f"trigger {idx}")
          cloudlog.info(f"DemoTriggers: fired [{label}] dist={dist:.0f}m prob={prob}")
          return True
        else:
          self._cooldown.add(idx)

      if not inside and idx in self._cooldown:
        self._cooldown.discard(idx)

    return False
