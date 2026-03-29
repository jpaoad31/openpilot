# IMU Bump Detection Fallback

## Problem

The bump detector (`selfdrive/ui/onroad/bump_detector.py`) relies on `carState.aEgo` for longitudinal acceleration. When no car is connected (e.g. bench testing, development on bare Comma 4 hardware), `carState.aEgo` is always 0 and bump detection is effectively disabled.

The device has an onboard IMU (LSM6DS3) that produces acceleration data regardless of car connection. This data should be used as a fallback.

## Proposed Behavior

- **Default:** Continue using `carState.aEgo` as the primary acceleration source. This is the calibrated longitudinal value and is preferred when available.
- **Fallback:** If `carState.aEgo` has been 0 for a sustained period (indicating no car connection or non-functional CAN), switch to the IMU accelerometer's x-axis (longitudinal) as the input to the bump detector.
- **Switch back:** If `carState.aEgo` becomes non-zero, revert to using it immediately.

## Detection Criteria

The fallback should be activated when:

1. `carState.aEgo == 0.0` for at least `N` consecutive frames (e.g. 50 frames / ~1 second)
2. The `accelerometer` message is valid and updating

This avoids false activation during brief moments where `aEgo` legitimately passes through zero (e.g. coasting).

## IMU Data Mapping

The IMU accelerometer (`sm['accelerometer'].acceleration`) reports 3-axis values:

| Axis | Direction | Device Orientation |
|------|-----------|-------------------|
| x | Longitudinal | Forward/backward along road |
| y | Lateral | Left/right |
| z | Vertical | Up/down (gravity) |

For bump detection:
- **Longitudinal bumps** (current behavior parity): use `acceleration.v[0]` (x-axis)
- **Vertical bumps** (better sensitivity): use `acceleration.v[2]` (z-axis), subtract gravity (~9.81 m/s²)

Vertical is more sensitive to road surface impacts, but the existing thresholds (`BUMP_ACCEL_THRESHOLD = 4.0 m/s²`, `BUMP_JERK_THRESHOLD = 10.0 m/s³`) were tuned for longitudinal. If using z-axis, thresholds may need recalibration.

### Recommendation

Use the z-axis (vertical) in fallback mode. Bumps manifest most clearly as vertical acceleration spikes. The gravity component is constant and can be subtracted with a simple high-pass filter or baseline offset.

## Changes Required

### `augmented_road_view.py`

In `_render()`, where the bump detector is currently called:

```python
# Current
bump_fired = self._bump_detector.update(ui_state.sm['carState'].aEgo)
```

Replace with logic that selects the acceleration source:

```python
a_ego = ui_state.sm['carState'].aEgo
accel_source = self._get_bump_accel(a_ego)
bump_fired = self._bump_detector.update(accel_source)
```

Add a helper method that tracks the zero-frame count and falls back to IMU:

```python
def _get_bump_accel(self, a_ego: float) -> float:
    if a_ego != 0.0:
        self._aego_zero_count = 0
        return a_ego

    self._aego_zero_count += 1
    if self._aego_zero_count < 50:
        return a_ego  # not enough zeros yet, still trust carState

    # Fallback to IMU z-axis, subtract gravity baseline
    accel = ui_state.sm['accelerometer'].acceleration
    return accel.v[2] - self._gravity_baseline
```

### Gravity baseline

Maintain a slow-moving average of the z-axis to estimate the gravity offset:

```python
# On each frame when IMU is active:
raw_z = sm['accelerometer'].acceleration.v[2]
self._gravity_baseline = 0.99 * self._gravity_baseline + 0.01 * raw_z
```

This converges to ~9.81 m/s² at rest and adapts to device mounting angle.

### Threshold Tuning

The existing thresholds were set for `carState.aEgo` (longitudinal, CAN-derived). IMU vertical acceleration from bumps tends to be sharper and larger in magnitude. Consider:

| Parameter | Current (aEgo) | IMU Fallback (suggested starting point) |
|-----------|----------------|----------------------------------------|
| `BUMP_ACCEL_THRESHOLD` | 4.0 m/s² | 3.0 m/s² |
| `BUMP_JERK_THRESHOLD` | 10.0 m/s³ | 8.0 m/s³ |

These should be validated empirically by shaking/tapping the device and observing the IMU monitor output.

### SubMaster Update

`augmented_road_view.py` does not currently subscribe to `accelerometer`. The `AugmentedRoadView` class or its parent needs to ensure `accelerometer` is in the SubMaster's service list when fallback mode is possible.

## Testing

1. **Bench test (no car):** Run `uiview.py` + `sensord`, tap/shake the device, confirm popup triggers
2. **In-car test:** Verify `carState.aEgo` is used (fallback never activates), existing bump detection behavior unchanged
3. **Transition test:** Start with no car, confirm IMU fallback works, then connect panda — verify it switches back to `carState.aEgo`

## Out of Scope

- Changing the bump detector class interface (it stays as `update(float) -> bool`)
- Using IMU data when carState is available
- Fusing IMU + CAN data
