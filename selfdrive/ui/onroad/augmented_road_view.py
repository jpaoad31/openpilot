import os
import time
import numpy as np
import pyray as rl
from cereal import log, messaging
from msgq.visionipc import VisionStreamType
from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.onroad.alert_renderer import AlertRenderer
from openpilot.selfdrive.ui.onroad.bump_detector import BumpDetector
from openpilot.selfdrive.ui.onroad.demo_triggers import DemoTriggers
from openpilot.selfdrive.ui.onroad.driver_state import DriverStateRenderer
from openpilot.selfdrive.ui.onroad.hazard_ahead_renderer import HazardAheadRenderer
from openpilot.selfdrive.ui.onroad.hazard_fetcher import HazardFetcher
from openpilot.selfdrive.ui.onroad.hazard_popup import HazardPopup
from openpilot.selfdrive.ui.onroad.hazard_detection_metrics import metrics as comma1_metrics
from openpilot.selfdrive.ui.onroad.hazard_reporter import HazardReporter
from openpilot.selfdrive.ui.onroad.hud_renderer import HudRenderer
from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
from openpilot.selfdrive.ui.onroad.cameraview import CameraView
from openpilot.system.ui.lib.application import gui_app
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.transformations.camera import DEVICE_CAMERAS, DeviceCameraConfig, view_frame_from_device_frame
from openpilot.selfdrive.ui.onroad.hazard_longitudinal_limits import cruise_kph_for_limits, longitudinal_limits_from_hazard
from openpilot.common.transformations.orientation import rot_from_euler

OpState = log.SelfdriveState.OpenpilotState
CALIBRATED = log.LiveCalibrationData.Status.calibrated
ROAD_CAM = VisionStreamType.VISION_STREAM_ROAD
WIDE_CAM = VisionStreamType.VISION_STREAM_WIDE_ROAD
DEFAULT_DEVICE_CAMERA = DEVICE_CAMERAS["tici", "ar0231"]

BORDER_COLORS = {
  UIStatus.DISENGAGED: rl.Color(0x12, 0x28, 0x39, 0xFF),  # Blue for disengaged state
  UIStatus.OVERRIDE: rl.Color(0x89, 0x92, 0x8D, 0xFF),  # Gray for override state
  UIStatus.ENGAGED: rl.Color(0x16, 0x7F, 0x40, 0xFF),  # Green for engaged state
}

WIDE_CAM_MAX_SPEED = 10.0  # m/s (22 mph)
ROAD_CAM_MIN_SPEED = 15.0  # m/s (34 mph)
INF_POINT = np.array([1000.0, 0.0, 0.0])

# Touch this file from SSH to manually trigger the hazard popup:
#   touch /tmp/hazard_trigger
HAZARD_TRIGGER_FILE = "/tmp/hazard_trigger"
_ROADPASS_LONGITUDINAL_PARAM = "RoadPassLongitudinalEnabled"


def _roadpass_longitudinal_param_enabled(params: Params) -> bool:
  try:
    return params.get_bool(_ROADPASS_LONGITUDINAL_PARAM)
  except UnknownKeyName:
    return False


class AugmentedRoadView(CameraView):
  def __init__(self, stream_type: VisionStreamType = VisionStreamType.VISION_STREAM_ROAD):
    super().__init__("camerad", stream_type)
    self._set_placeholder_color(BORDER_COLORS[UIStatus.DISENGAGED])

    self.device_camera: DeviceCameraConfig | None = None
    self.view_from_calib = view_frame_from_device_frame.copy()
    self.view_from_wide_calib = view_frame_from_device_frame.copy()

    self._matrix_cache_key = (0, 0.0, 0.0, stream_type)
    self._cached_matrix: np.ndarray | None = None
    self._content_rect = rl.Rectangle()

    self.model_renderer = ModelRenderer()
    self._hud_renderer = HudRenderer()
    self.alert_renderer = AlertRenderer()
    self.driver_state_renderer = DriverStateRenderer()

    self._bump_detector = BumpDetector()
    self._demo_triggers = DemoTriggers()
    self._hazard_popup = HazardPopup()
    self._hazard_reporter = HazardReporter()
    self._hazard_fetcher = HazardFetcher()
    self._hazard_ahead_renderer = HazardAheadRenderer(self._hazard_fetcher)

    # hazard_ids that were auto-confirmed via bump near a known hazard
    self._confirmed_hazard_ids: set[str] = set()

    self._params = Params()
    self._pm = messaging.PubMaster(['uiDebug'])

  def _render(self, rect):
    # Only render when system is started to avoid invalid data access
    start_draw = time.monotonic()
    if not ui_state.started:
      return

    self._switch_stream_if_needed(ui_state.sm)

    # Update calibration before rendering
    self._update_calibration()

    # Create inner content area with border padding
    self._content_rect = rl.Rectangle(
      rect.x + UI_BORDER_SIZE,
      rect.y + UI_BORDER_SIZE,
      rect.width - 2 * UI_BORDER_SIZE,
      rect.height - 2 * UI_BORDER_SIZE,
    )

    # Enable scissor mode to clip all rendering within content rectangle boundaries
    # This creates a rendering viewport that prevents graphics from drawing outside the border
    rl.begin_scissor_mode(
      int(self._content_rect.x),
      int(self._content_rect.y),
      int(self._content_rect.width),
      int(self._content_rect.height)
    )

    # Render the base camera view
    super()._render(rect)

    # Feed current GPS into the background hazard fetcher
    gps = ui_state.sm['gpsLocationExternal']
    self._hazard_fetcher.update_gps(
      gps.latitude, gps.longitude, gps.bearingDeg,
      ui_state.sm['carState'].vEgo, gps.hasFix,
    )

    # Draw all UI overlays
    self.model_renderer.render(self._content_rect)
    self._hud_renderer.render(self._content_rect)
    self.alert_renderer.render(self._content_rect)
    self.driver_state_renderer.render(self._content_rect)
    self._hazard_ahead_renderer.render(self._content_rect)

    # Show hazard popup on bump detection, demo geofence trigger, or manual SSH trigger
    bump_fired = self._bump_detector.update(ui_state.sm['carState'].aEgo)
    demo_fired = not bump_fired and gps.hasFix and self._demo_triggers.check(gps.latitude, gps.longitude)
    manual_fired = not bump_fired and not demo_fired and os.path.exists(HAZARD_TRIGGER_FILE)
    if manual_fired:
      os.remove(HAZARD_TRIGGER_FILE)

    hazard_detected = bump_fired or demo_fired
    if hazard_detected:
      if bump_fired:
        diag = self._bump_detector.consume_last_trigger_diag() or {}
        comma1_metrics.record_bump_trigger(diag)
        trigger_source = "bump_detector"
      else:
        trigger_source = "demo_trigger"

      # Check if this matches a known hazard within 50m
      nearby = self._find_nearby_known_hazard(gps)
      if nearby is not None and nearby.device_previously_reported:
        # Returning user hitting the same hazard — auto-confirm, no popup
        self._hazard_reporter.confirm(nearby.hazard_id, gps.latitude, gps.longitude)
        self._confirmed_hazard_ids.add(nearby.hazard_id)
        comma1_metrics.record_auto_confirm()
      else:
        # New hazard, or first time this device encounters a known one — show popup
        self._show_hazard_popup(gps, trigger_source)
    elif manual_fired:
      comma1_metrics.record_manual_trigger()
      self._show_hazard_popup(gps, "manual")

    # Send confirm/clear for hazards the device visited (< 30m) then departed (> 100m)
    for hazard_id in self._hazard_ahead_renderer.consume_departed():
      if hazard_id in self._confirmed_hazard_ids:
        pass  # already auto-confirmed via bump match
      elif gps.hasFix:
        self._hazard_reporter.clear(hazard_id, gps.latitude, gps.longitude)
        comma1_metrics.record_auto_clear()

    # End clipping region
    rl.end_scissor_mode()

    # Draw colored border based on driving state
    self._draw_border(rect)

    # publish uiDebug (draw timing + optional RoadPass longitudinal hints for plannerd)
    msg = messaging.new_message('uiDebug')
    dbg = msg.uiDebug
    dbg.drawTimeMillis = (time.monotonic() - start_draw) * 1000
    dbg.roadPassLongitudinalActive = False
    dbg.roadPassHazardDistanceM = 0.0
    dbg.roadPassHazardMaxSpeedMs = 0.0
    dbg.roadPassHazardMaxAccelMs2 = 0.0
    if _roadpass_longitudinal_param_enabled(self._params) and gps.hasFix:
      ahead = self._hazard_ahead_renderer.get_active_ahead_hazard(
        gps.latitude, gps.longitude, gps.bearingDeg, ui_state.sm['carState'].vEgo,
      )
      if ahead is not None:
        _, dist_m, warn_m = ahead
        v_kph = cruise_kph_for_limits(ui_state.sm['carState'])
        v_cap, a_cap = longitudinal_limits_from_hazard(v_kph, dist_m, warn_m)
        dbg.roadPassLongitudinalActive = True
        dbg.roadPassHazardDistanceM = float(dist_m)
        dbg.roadPassHazardMaxSpeedMs = float(v_cap)
        dbg.roadPassHazardMaxAccelMs2 = float(a_cap)
    self._pm.send('uiDebug', msg)

  def _find_nearby_known_hazard(self, gps):
    """Return the nearest fetched hazard within 50m, or None."""
    if not gps.hasFix:
      return None
    for h in self._hazard_fetcher.get_hazards():
      if h.distance_m(gps.latitude, gps.longitude) < 50:
        return h
    return None

  def _show_hazard_popup(self, gps, trigger_source: str):
    if gui_app.widget_in_stack(self._hazard_popup):
      return
    self._hazard_reporter.detect(ui_state.sm, trigger_source)
    self._hazard_popup.set_response_callback(self._hazard_reporter.respond)
    gui_app.push_widget(self._hazard_popup)

  def _handle_mouse_press(self, _):
    if not self._hud_renderer.user_interacting() and self._click_callback is not None:
      self._click_callback()

  def _handle_mouse_release(self, _):
    # We only call click callback on press if not interacting with HUD
    pass

  def _draw_border(self, rect: rl.Rectangle):
    rl.draw_rectangle_lines_ex(rect, UI_BORDER_SIZE, rl.BLACK)
    border_roundness = 0.12
    border_color = BORDER_COLORS.get(ui_state.status, BORDER_COLORS[UIStatus.DISENGAGED])
    border_rect = rl.Rectangle(rect.x + UI_BORDER_SIZE, rect.y + UI_BORDER_SIZE,
                               rect.width - 2 * UI_BORDER_SIZE, rect.height - 2 * UI_BORDER_SIZE)
    rl.draw_rectangle_rounded_lines_ex(border_rect, border_roundness, 10, UI_BORDER_SIZE, border_color)

  def _switch_stream_if_needed(self, sm):
    if sm['selfdriveState'].experimentalMode and WIDE_CAM in self.available_streams:
      v_ego = sm['carState'].vEgo
      if v_ego < WIDE_CAM_MAX_SPEED:
        target = WIDE_CAM
      elif v_ego > ROAD_CAM_MIN_SPEED:
        target = ROAD_CAM
      else:
        # Hysteresis zone - keep current stream
        target = self.stream_type
    else:
      target = ROAD_CAM

    if self.stream_type != target:
      self.switch_stream(target)

  def _update_calibration(self):
    # Update device camera if not already set
    sm = ui_state.sm
    if not self.device_camera and sm.seen['roadCameraState'] and sm.seen['deviceState']:
      self.device_camera = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['roadCameraState'].sensor))]

    # Check if live calibration data is available and valid
    if not (sm.updated["liveCalibration"] and sm.valid['liveCalibration']):
      return

    calib = sm['liveCalibration']
    if len(calib.rpyCalib) != 3 or calib.calStatus != CALIBRATED:
      return

    # Update view_from_calib matrix
    device_from_calib = rot_from_euler(calib.rpyCalib)
    self.view_from_calib = view_frame_from_device_frame @ device_from_calib

    # Update wide calibration if available
    if hasattr(calib, 'wideFromDeviceEuler') and len(calib.wideFromDeviceEuler) == 3:
      wide_from_device = rot_from_euler(calib.wideFromDeviceEuler)
      self.view_from_wide_calib = view_frame_from_device_frame @ wide_from_device @ device_from_calib

  def _calc_frame_matrix(self, rect: rl.Rectangle) -> np.ndarray:
    # Check if we can use cached matrix
    cache_key = (
      ui_state.sm.recv_frame['liveCalibration'],
      self._content_rect.width,
      self._content_rect.height,
      self.stream_type
    )
    if cache_key == self._matrix_cache_key and self._cached_matrix is not None:
      return self._cached_matrix

    # Get camera configuration
    device_camera = self.device_camera or DEFAULT_DEVICE_CAMERA
    is_wide_camera = self.stream_type == WIDE_CAM
    intrinsic = device_camera.ecam.intrinsics if is_wide_camera else device_camera.fcam.intrinsics
    calibration = self.view_from_wide_calib if is_wide_camera else self.view_from_calib
    zoom = 2.0 if is_wide_camera else 1.1

    # Calculate transforms for vanishing point
    calib_transform = intrinsic @ calibration
    kep = calib_transform @ INF_POINT

    # Calculate center points and dimensions
    x, y = self._content_rect.x, self._content_rect.y
    w, h = self._content_rect.width, self._content_rect.height
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Calculate max allowed offsets with margins
    margin = 5
    max_x_offset = cx * zoom - w / 2 - margin
    max_y_offset = cy * zoom - h / 2 - margin

    # Calculate and clamp offsets to prevent out-of-bounds issues
    try:
      if abs(kep[2]) > 1e-6:
        x_offset = np.clip((kep[0] / kep[2] - cx) * zoom, -max_x_offset, max_x_offset)
        y_offset = np.clip((kep[1] / kep[2] - cy) * zoom, -max_y_offset, max_y_offset)
      else:
        x_offset, y_offset = 0, 0
    except (ZeroDivisionError, OverflowError):
      x_offset, y_offset = 0, 0

    # Cache the computed transformation matrix to avoid recalculations
    self._matrix_cache_key = cache_key
    self._cached_matrix = np.array([
      [zoom * 2 * cx / w, 0, -x_offset / w * 2],
      [0, zoom * 2 * cy / h, -y_offset / h * 2],
      [0, 0, 1.0]
    ])

    video_transform = np.array([
      [zoom, 0.0, (w / 2 + x - x_offset) - (cx * zoom)],
      [0.0, zoom, (h / 2 + y - y_offset) - (cy * zoom)],
      [0.0, 0.0, 1.0]
    ])
    self.model_renderer.set_transform(video_transform @ calib_transform)

    return self._cached_matrix


if __name__ == "__main__":
  gui_app.init_window("OnRoad Camera View")
  road_camera_view = AugmentedRoadView(ROAD_CAM)
  gui_app.push_widget(road_camera_view)
  print("***press space to switch camera view***")
  try:
    for _ in gui_app.render():
      ui_state.update()
      if rl.is_key_released(rl.KeyboardKey.KEY_SPACE):
        if WIDE_CAM in road_camera_view.available_streams:
          stream = ROAD_CAM if road_camera_view.stream_type == WIDE_CAM else WIDE_CAM
          road_camera_view.switch_stream(stream)
  finally:
    road_camera_view.close()
