"""
Crowd scoring for RoadPass hazards (client-side, from API counts).

The /hazards/ahead endpoint returns three counts per hazard:
  - report_count:  how many distinct detection events matched this location
  - confirm_count: warned devices that later confirmed "yes it's still there"
  - reject_count:  warned devices that said "it's clear now"

Score formula:
  If (confirm_count + reject_count) > 0:
    score_pct = round(100 * confirm_count / (confirm_count + reject_count))
  Else:
    score_pct derived from report_count alone (more reports → higher score)

Tiers:
  high:  score_pct ≥ 70 and (confirm_count + reject_count) ≥ 2, or report_count ≥ 3
  med:   score_pct ≥ 40 or (confirm_count + reject_count) ≥ 3, or report_count ≥ 2
  low:   otherwise

If there is no usable data, hazard_score_from_api returns None.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HazardScore:
  score_pct: int
  tier: str  # "high" | "med" | "low"
  report_count: int
  confirm_count: int
  reject_count: int

  @property
  def tier_label(self) -> str:
    return {"high": "High", "med": "Medium", "low": "Low"}.get(self.tier, self.tier)


def _tier(score_pct: int, responded: int, report_count: int) -> str:
  if report_count >= 3 or (score_pct >= 70 and responded >= 2):
    return "high"
  if report_count >= 2 or score_pct >= 40 or responded >= 3:
    return "med"
  return "low"


def hazard_score_from_api(hazard_dict: dict) -> HazardScore | None:
  """Build a HazardScore from one element of GET /hazards/ahead JSON hazards[]."""
  report_count = int(hazard_dict.get("report_count", 0) or 0)
  confirm_count = int(hazard_dict.get("confirm_count", 0) or 0)
  reject_count = int(hazard_dict.get("reject_count", 0) or 0)
  responded = confirm_count + reject_count

  if responded <= 0 and report_count <= 0:
    return None

  if responded > 0:
    score_pct = int(round(100.0 * confirm_count / responded))
  elif report_count > 0:
    # No confirm/reject data yet — use report_count as a rough confidence proxy.
    # 1 report → 50%, 2 → 70%, 3+ → 85%
    score_pct = min(85, 35 + report_count * 17)
  else:
    score_pct = 0

  score_pct = max(0, min(100, score_pct))
  tier = _tier(score_pct, responded, report_count)
  return HazardScore(
    score_pct=score_pct,
    tier=tier,
    report_count=report_count,
    confirm_count=confirm_count,
    reject_count=reject_count,
  )
