import pyray as rl

from openpilot.selfdrive.ui.onroad.hazard_ahead_renderer import HazardAheadRenderer
from openpilot.selfdrive.ui.onroad.hazard_fetcher import HazardAhead, HazardFetcher
from openpilot.selfdrive.ui.mici.onroad import SIDE_PANEL_WIDTH
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets.label import UnifiedLabel

# Gradient alert style matching the "Changing Lanes" alerts
ALERT_MARGIN = 18
ICON_MARGIN_X = 8
ICON_MARGIN_Y = 5
ALERT_BG_COLOR = rl.Color(255, 115, 0, 255)  # orange, matches AlertStatus.userPrompt


class MiciHazardAheadRenderer(HazardAheadRenderer):
  """
  Mici-sized hazard ahead warning. Uses the same gradient overlay style as the
  lane change alerts — a gradient from top with an icon on the left and text
  beside it. Inherits lifecycle logic from the big UI HazardAheadRenderer.
  """

  def __init__(self, fetcher: HazardFetcher):
    super().__init__(fetcher)
    self._txt_warning_icon = gui_app.texture('icons_mici/exclamation_point.png', 80, 80)
    self._main_label = UnifiedLabel(text="", font_size=60, font_weight=FontWeight.DISPLAY,
                                    line_height=0.86, letter_spacing=-0.02)
    self._sub_label = UnifiedLabel(text="", font_size=36, font_weight=FontWeight.ROMAN,
                                   line_height=0.86, letter_spacing=0.025)

  def _draw_card(self, rect: rl.Rectangle, hazard: HazardAhead, distance_m: float, passed: bool = False) -> None:
    # Alert height: ~58% of content rect, matching lane change small alerts
    bg_height = round(rect.height * 0.583)
    alpha = 0.90

    # Draw gradient background from top
    color = rl.Color(ALERT_BG_COLOR.r, ALERT_BG_COLOR.g, ALERT_BG_COLOR.b, int(255 * alpha))
    translucent = rl.Color(ALERT_BG_COLOR.r, ALERT_BG_COLOR.g, ALERT_BG_COLOR.b, 0)

    solid_height = round(bg_height * 0.2)
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), solid_height, color)
    rl.draw_rectangle_gradient_v(
      int(rect.x), int(rect.y + solid_height),
      int(rect.width), int(bg_height - solid_height),
      color, translucent,
    )

    # Draw warning icon on the left
    icon_x = int(rect.x + ICON_MARGIN_X)
    icon_y = int(rect.y + ICON_MARGIN_Y)
    rl.draw_texture_ex(self._txt_warning_icon, rl.Vector2(icon_x, icon_y), 0.0, 1.0, rl.WHITE)

    # Text area: right of the icon
    text_x = rect.x + ICON_MARGIN_X + self._txt_warning_icon.width + ALERT_MARGIN
    text_width = rect.width - text_x + rect.x - SIDE_PANEL_WIDTH

    # Main text
    if passed:
      main_text = "hazard passed"
    else:
      main_text = f"hazard ahead - {int(distance_m)}m"

    text_color = rl.Color(255, 255, 255, int(255 * 0.9))
    text_rect = rl.Rectangle(text_x, rect.y - 4, text_width, rect.height)
    self._main_label.set_text(main_text)
    self._main_label.set_text_color(text_color)
    self._main_label.render(text_rect)

    # Sub text (confidence)
    if hazard.score is not None:
      sub_text = f"{hazard.score.tier_label.lower()} confidence - {hazard.score.score_pct}%"
    else:
      sub_text = ""

    if sub_text:
      main_h = self._main_label.get_content_height(int(text_width))
      sub_y = self._main_label.rect.y + main_h - 4
      sub_color = rl.Color(255, 255, 255, int(255 * 0.65))
      sub_rect = rl.Rectangle(text_x, sub_y, text_width, rect.height - sub_y)
      self._sub_label.set_text(sub_text)
      self._sub_label.set_text_color(sub_color)
      self._sub_label.render(sub_rect)
