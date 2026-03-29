import time
import pyray as rl
from collections.abc import Callable
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialogBase
from openpilot.selfdrive.ui.onroad.hazard_sounds import play_hazard_detected
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle

POPUP_TIMEOUT = 15.0

# Scaled for 536×240 mici screen
POPUP_WIDTH = 440
POPUP_HEIGHT = 180
MARGIN = 16
BUTTON_HEIGHT = 55
BUTTON_FONT = 28
TIMER_BAR_HEIGHT = 8
TITLE_FONT_SIZE = 50

PANEL_COLOR = rl.Color(20, 20, 20, 240)
TIMER_BAR_BG = rl.Color(70, 70, 70, 255)
TIMER_BAR_FILL = rl.Color(230, 140, 30, 255)
BORDER_RADIUS = 12


class MiciHazardPopup(BigDialogBase):
  """
  Full-screen hazard confirmation popup for the mici (Comma 4) UI.
  Matches BigDialogBase conventions (full-screen, swipe-down dismiss).
  """

  def __init__(self):
    super().__init__()
    self._response_callback: Callable[[str, float], None] | None = None
    self._start_time: float = 0.0

    self._no_button = self._child(Button("No", self._handle_no, font_size=BUTTON_FONT, button_style=ButtonStyle.NORMAL))
    self._yes_button = self._child(Button("Yes", self._handle_yes, font_size=BUTTON_FONT, button_style=ButtonStyle.DANGER))

  def set_response_callback(self, callback: Callable[[str, float], None] | None) -> None:
    self._response_callback = callback

  def show_event(self):
    super().show_event()
    self._start_time = time.monotonic()
    play_hazard_detected()

  def _elapsed(self) -> float:
    return time.monotonic() - self._start_time

  def _fire_response(self, answer: str) -> None:
    if self._response_callback is not None:
      self._response_callback(answer, self._elapsed())

  def _handle_yes(self):
    self._fire_response("yes")
    gui_app.pop_widget()

  def _handle_no(self):
    self._fire_response("no")
    gui_app.pop_widget()

  def _render(self, rect: rl.Rectangle):
    elapsed = self._elapsed()
    progress = max(0.0, 1.0 - elapsed / POPUP_TIMEOUT)

    if progress <= 0.0:
      self._fire_response("timeout")
      gui_app.pop_widget()
      return

    # Full-screen black background (mici dialog convention)
    rl.draw_rectangle_rec(rect, rl.Color(0, 0, 0, 200))

    # Centered panel
    px = rect.x + (rect.width - POPUP_WIDTH) / 2
    py = rect.y + (rect.height - POPUP_HEIGHT) / 2
    panel = rl.Rectangle(px, py, POPUP_WIDTH, POPUP_HEIGHT)

    roundness = BORDER_RADIUS / (min(POPUP_WIDTH, POPUP_HEIGHT) / 2)
    rl.draw_rectangle_rounded(panel, roundness, 10, PANEL_COLOR)

    # Title
    font = gui_app.font(FontWeight.BOLD)
    text = "Hazard"
    text_size = measure_text_cached(font, text, TITLE_FONT_SIZE)
    content_bottom = py + POPUP_HEIGHT - MARGIN - BUTTON_HEIGHT - MARGIN - TIMER_BAR_HEIGHT - MARGIN
    title_y = py + MARGIN + (content_bottom - py - MARGIN - text_size.y) / 2
    title_x = px + (POPUP_WIDTH - text_size.x) / 2
    rl.draw_text_ex(font, text, rl.Vector2(int(title_x), int(title_y)), TITLE_FONT_SIZE, 0, rl.WHITE)

    # Buttons
    btn_y = py + POPUP_HEIGHT - MARGIN - TIMER_BAR_HEIGHT - MARGIN - BUTTON_HEIGHT
    btn_w = (POPUP_WIDTH - 3 * MARGIN) / 2
    self._no_button.render(rl.Rectangle(px + MARGIN, btn_y, btn_w, BUTTON_HEIGHT))
    self._yes_button.render(rl.Rectangle(px + POPUP_WIDTH - btn_w - MARGIN, btn_y, btn_w, BUTTON_HEIGHT))

    # Timer bar
    bar_x = px + MARGIN
    bar_y = py + POPUP_HEIGHT - MARGIN - TIMER_BAR_HEIGHT
    bar_w = POPUP_WIDTH - 2 * MARGIN
    rl.draw_rectangle_rounded(rl.Rectangle(bar_x, bar_y, bar_w, TIMER_BAR_HEIGHT), 0.5, 10, TIMER_BAR_BG)
    fill_w = bar_w * progress
    if fill_w >= 2:
      rl.draw_rectangle_rounded(rl.Rectangle(bar_x, bar_y, fill_w, TIMER_BAR_HEIGHT), 0.5, 10, TIMER_BAR_FILL)
