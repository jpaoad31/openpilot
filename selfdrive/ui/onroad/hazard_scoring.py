"""
Crowd scoring for RoadPass hazards — server-computed.

The /hazards/ahead endpoint now returns confidence_score (0–100) and
confidence_tier ("high" | "medium" | "low") directly. The device just reads
and displays them; all scoring logic lives server-side.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HazardScore:
  score_pct: int
  tier: str  # "high" | "medium" | "low"

  @property
  def tier_label(self) -> str:
    return {"high": "High", "medium": "Medium", "low": "Low"}.get(self.tier, self.tier)


def hazard_score_from_api(hazard_dict: dict) -> HazardScore | None:
  """Read the server-computed confidence score and tier from a hazard object."""
  tier = hazard_dict.get("confidence_tier")
  score = hazard_dict.get("confidence_score")
  if tier is None and score is None:
    return None
  return HazardScore(
    score_pct=max(0, min(100, int(score or 0))),
    tier=str(tier or "low"),
  )
