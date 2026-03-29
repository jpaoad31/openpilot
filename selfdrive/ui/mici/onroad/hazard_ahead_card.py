import pyray as rl

from openpilot.selfdrive.ui.onroad.hazard_ahead_renderer import HazardAheadRenderer
from openpilot.selfdrive.ui.onroad.hazard_fetcher import HazardAhead, HazardFetcher
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Scaled for 536×240 mici screen
CARD_WIDTH = 240
CARD_HEIGHT = 50
CARD_MARGIN = 8
CARD_RADIUS = 8
FONT_SIZE_MAIN = 22
FONT_SIZE_SUB = 16

CARD_BG = rl.Color(170, 85, 0, 230)


class MiciHazardAheadRenderer(HazardAheadRenderer):
  """
  Mici-sized hazard ahead card. Inherits all lifecycle logic (visit/depart,
  suppression, timeout) from the big UI HazardAheadRenderer and only overrides
  the drawing to fit the 536×240 screen.
  """

  def __init__(self, fetcher: HazardFetcher):
    super().__init__(fetcher)

  def _draw_card(self, rect: rl.Rectangle, hazard: HazardAhead, distance_m: float, passed: bool = False) -> None:
    # Position top-right of the camera area, left of the side panel
    x = rect.x + rect.width - SIDE_PANEL_WIDTH - CARD_WIDTH - CARD_MARGIN
    y = rect.y + CARD_MARGIN
    card_rect = rl.Rectangle(x, y, CARD_WIDTH, CARD_HEIGHT)

    roundness = CARD_RADIUS / (min(CARD_WIDTH, CARD_HEIGHT) / 2)
    rl.draw_rectangle_rounded(card_rect, roundness, 10, CARD_BG)

    if passed:
      line_main = "Hazard passed"
    else:
      line_main = f"Hazard ahead \xb7 {int(distance_m)}m"

    if hazard.score is not None:
      sc = hazard.score
      line_sub = f"{sc.tier_label} \xb7 {sc.score_pct}%"
    else:
      line_sub = ""

    font = gui_app.font(FontWeight.BOLD)
    sz_m = measure_text_cached(font, line_main, FONT_SIZE_MAIN)
    gap = 2
    block_h = sz_m.y + (gap + measure_text_cached(font, line_sub, FONT_SIZE_SUB).y if line_sub else 0)
    ty0 = y + (CARD_HEIGHT - block_h) / 2

    tx_m = x + (CARD_WIDTH - sz_m.x) / 2
    rl.draw_text_ex(font, line_main, rl.Vector2(int(tx_m), int(ty0)), FONT_SIZE_MAIN, 0, rl.WHITE)

    if line_sub:
      sz_s = measure_text_cached(font, line_sub, FONT_SIZE_SUB)
      tx_s = x + (CARD_WIDTH - sz_s.x) / 2
      rl.draw_text_ex(font, line_sub, rl.Vector2(int(tx_s), int(ty0 + sz_m.y + gap)), FONT_SIZE_SUB, 0, rl.WHITE)
