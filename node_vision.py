from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math
import threading
import time

# ── Camera specifications (Raspberry Pi Cam V2) ────────────────────────────────
FOV_DEG    = 62.2          # Horizontal field of view in degrees
RES_WIDTH  = 640           # Frame width in pixels
RES_HEIGHT = 480           # Frame height in pixels
CENTER_X   = RES_WIDTH / 2.0

# ── Colour filter (HSV) ────────────────────────────────────────────────────────
# Tune these values with a colour-picker / slider script for your lighting.
LOWER_ORANGE = (5,  120, 100)
UPPER_ORANGE = (25, 255, 255)

# ── Detection thresholds ───────────────────────────────────────────────────────
DEADZONE_PIXELS = 40   # Centre dead-zone width (pixels) before steering
MIN_RADIUS      = 5    # Minimum ball radius in pixels (filters noise)

# ── Ball physical size ─────────────────────────────────────────────────────────
BALL_RADIUS_MM = 21.0  # Real-world ball radius in mm (used for distance calc)

# ── Robot geometry ────────────────────────────────────────────────────────────
ROBOT_RADIUS = 0.09   # metres — camera sits one radius ahead of the robot centre

# ── Broker key ────────────────────────────────────────────────────────────────
BROKER_KEY = "ball"

SIM_REPLACE = False  # Set True to force simulated ball even if a camera is found

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_vision", broker=mb, print_every=100)

_robot_pos = None   # (x, y) metres — from robot_position broker key
_imu_pitch = None   # degrees       — from imu_pitch broker key

_hw_available = False
try:
    import cv2
    import numpy as np
    _hw_available = True
except ImportError as e:
    print(f"[VISION] OpenCV not available ({e}) — node will not run.")


def _process_frame(frame):
    """
    Detect the orange ball in a BGR frame.

    Returns a dict:
        command      – "FORWARD" | "LEFT" | "RIGHT" | "NO_BALL"
        distance_cm  – estimated distance in centimetres (0.0 if no ball)
        x_center     – detected pixel x-centre (-1 if no ball)
        radius       – detected pixel radius (-1 if no ball)
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array(LOWER_ORANGE),
                       np.array(UPPER_ORANGE))

    # Noise reduction: erode then dilate
    mask = cv2.erode(mask,  None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    largest = max(contours, key=cv2.contourArea)
    (x_center, _), radius = cv2.minEnclosingCircle(largest)

    if radius <= MIN_RADIUS:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    # ── Distance estimate via angular diameter ─────────────────────────────────
    # The ball subtends an angle of Ow_deg degrees in the image.
    # Using: distance = ball_radius / tan(half_angle)
    diameter_px = radius * 2.0
    Ow_deg = (FOV_DEG / RES_WIDTH) * diameter_px
    Ow_rad = math.radians(Ow_deg)
    distance_cm = (BALL_RADIUS_MM / math.tan(Ow_rad / 2.0)) / 10.0 if Ow_rad > 0 else 0.0

    # ── Angle to ball (horizontal) ─────────────────────────────────────────────
    # Map pixel offset from centre to degrees using the horizontal FOV.
    # Positive = ball is to the right, negative = ball is to the left.
    error_x   = x_center - CENTER_X
    angle_deg = (error_x / RES_WIDTH) * FOV_DEG

    # ── Steering command ───────────────────────────────────────────────────────
    if error_x < -DEADZONE_PIXELS:
        command = "LEFT"
    elif error_x > DEADZONE_PIXELS:
        command = "RIGHT"
    else:
        command = "FORWARD"

    return {
        "command":     command,
        "distance_cm": round(distance_cm, 1),
        "angle_deg":   round(angle_deg, 1),
        "x_center":    round(float(x_center), 1),
        "radius":      round(float(radius), 1),
    }


class _SimBall:
    """
    Physics-based ball simulation that renders synthetic BGR camera frames.

    The ball moves in 3-D camera space (x_mm horizontal, z_mm depth) and
    bounces off the depth limits and the horizontal FOV boundary.  Each call
    to render() advances the simulation by the elapsed wall-clock time and
    returns a frame that _process_frame() can analyse exactly like a real one.
    """

    DIST_MIN_MM  =  300.0   # nearest the ball may come to the camera
    DIST_MAX_MM  = 2000.0   # furthest the ball may be from the camera
    SPEED_MM_S   =  300.0   # initial speed magnitude
    # Keep the ball within this fraction of the half-FOV to avoid clipping
    FOV_MARGIN   =  0.85

    def __init__(self):
        import random
        rng = random.Random()

        # Focal length in pixels — inverse of the degrees-per-pixel ratio
        self._focal_px = (RES_WIDTH / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))

        # Initial 3-D position
        self._z = rng.uniform(self.DIST_MIN_MM, self.DIST_MAX_MM)
        horiz_limit = math.tan(math.radians(FOV_DEG / 2.0)) * self._z * self.FOV_MARGIN
        self._x = rng.uniform(-horiz_limit, horiz_limit)

        # Initial velocity in a random direction
        angle = rng.uniform(0, 2 * math.pi)
        self._vx = math.cos(angle) * self.SPEED_MM_S
        self._vz = math.sin(angle) * self.SPEED_MM_S

        self._last_t = time.monotonic()

        # Pre-compute a solid orange BGR colour that lies within LOWER/UPPER_ORANGE.
        # HSV(15, 200, 220) is safely mid-range for both hue and saturation.
        _hsv = np.array([[[15, 200, 220]]], dtype=np.uint8)
        bgr  = cv2.cvtColor(_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        self._orange_bgr = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    def render(self):
        """Advance physics and return a synthetic BGR frame."""
        now = time.monotonic()
        dt  = now - self._last_t
        self._last_t = now

        # Advance position
        self._x += self._vx * dt
        self._z += self._vz * dt

        # Bounce off depth limits
        if self._z < self.DIST_MIN_MM:
            self._z  = self.DIST_MIN_MM
            self._vz = abs(self._vz)
        elif self._z > self.DIST_MAX_MM:
            self._z  = self.DIST_MAX_MM
            self._vz = -abs(self._vz)

        # Bounce off horizontal FOV boundary (recomputed at current depth)
        horiz_limit = math.tan(math.radians(FOV_DEG / 2.0)) * self._z * self.FOV_MARGIN
        if self._x < -horiz_limit:
            self._x  = -horiz_limit
            self._vx = abs(self._vx)
        elif self._x > horiz_limit:
            self._x  = horiz_limit
            self._vx = -abs(self._vx)

        # Project 3-D position onto the virtual sensor
        px       = int(CENTER_X + (self._x / self._z) * self._focal_px)
        radius_px = max(1, int((BALL_RADIUS_MM / self._z) * self._focal_px))

        # Render: black background with one filled orange circle
        frame = np.zeros((RES_HEIGHT, RES_WIDTH, 3), dtype=np.uint8)
        cv2.circle(frame, (px, RES_HEIGHT // 2), radius_px, self._orange_bgr, -1)
        return frame


def _on_broker_update(key, value):
    global _robot_pos, _imu_pitch
    if value is None:
        return
    if key == "robot_position":
        try:
            p = json.loads(value)
            _robot_pos = (float(p["x"]), float(p["y"]))
        except Exception:
            pass
    elif key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except Exception:
            pass


def _compute_global_pos(distance_cm, angle_deg):
    """
    Project the detected ball into the global field frame.

    The camera is mounted one robot radius ahead of the robot centre,
    facing in the heading direction.  The ball bearing combines the
    heading with the camera's measured horizontal angle to the ball.
    Returns {"x": float, "y": float} or None if position/heading unknown.
    """
    if _robot_pos is None or _imu_pitch is None:
        return None
    rx, ry      = _robot_pos
    heading_rad = math.radians(_imu_pitch)
    cam_x = rx + ROBOT_RADIUS * math.cos(heading_rad)
    cam_y = ry + ROBOT_RADIUS * math.sin(heading_rad)
    bearing = heading_rad + math.radians(angle_deg)
    dist_m  = distance_cm / 100.0
    return {
        "x": round(cam_x + dist_m * math.cos(bearing), 3),
        "y": round(cam_y + dist_m * math.sin(bearing), 3),
    }


if __name__ == "__main__":
    if not _hw_available:
        raise SystemExit("[VISION] Cannot start: OpenCV is not installed.")

    # Seed broker values and start receiver thread
    try:
        val = mb.get("robot_position")
        if val is not None:
            p = json.loads(val)
            _robot_pos = (float(p["x"]), float(p["y"]))
    except Exception:
        pass
    try:
        val = mb.get("imu_pitch")
        if val is not None:
            _imu_pitch = float(val)
    except Exception:
        pass
    mb.setcallback(["robot_position", "imu_pitch"], _on_broker_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    print("[VISION] Starting headless vision system...")

    if SIM_REPLACE:
        cap = None
        sim = _SimBall()
        print("[VISION] SIM_REPLACE=True — using simulated ball.")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RES_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)

        if cap.isOpened():
            print(f"[VISION] Camera opened ({RES_WIDTH}x{RES_HEIGHT}). "
                  "Running detection loop (Ctrl+C to stop)...")
            sim = None
        else:
            cap.release()
            cap = None
            sim = _SimBall()
            print("[VISION] Camera not available — falling back to simulated ball.")

    last_log_time = time.time()

    try:
        while True:
            if sim is not None:
                frame = sim.render()
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[VISION] ERROR: Camera connection lost!")
                    break

            with _perf.measure("frame"):
                result = _process_frame(frame)
                if result["command"] != "NO_BALL":
                    result["global_pos"] = _compute_global_pos(
                        result["distance_cm"], result["angle_deg"]
                    )
                else:
                    result["global_pos"] = None
                mb.set(BROKER_KEY, json.dumps(result))

            # Log to console at most once per second
            now = time.time()
            if now - last_log_time >= 1.0:
                if result["command"] == "NO_BALL":
                    print("[VISION] Status: NO BALL")
                else:
                    print(f"[VISION] Status: {result['command']} | "
                          f"Distance: {result['distance_cm']:.1f} cm | "
                          f"Angle: {result['angle_deg']:.1f} deg")
                last_log_time = now

    except KeyboardInterrupt:
        print("\n[VISION] Stopped by user.")
    finally:
        if cap is not None:
            cap.release()
        mb.close()
        print("[VISION] Camera closed. System stopped.")
