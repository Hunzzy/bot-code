import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import time
import math
import numpy as np
from robus_core.libs.lib_telemtrybroker import TelemetryBroker

try:
    import board
    import busio
    import digitalio
    from adafruit_bno08x import (
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_LINEAR_ACCELERATION,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_ROTATION_VECTOR,
        BNO_REPORT_GRAVITY,
        PacketError,
    )
    from adafruit_bno08x.i2c import BNO08X_I2C
except ImportError as e:
    print(f"[IMU] Hardware libraries not available ({e}) — is this running on the robot? Exiting.")
    sys.exit(0)

# ── Configuration ──────────────────────────────────────────────────────────────
I2C_ADDRESS          = 0x4b
RESET_PIN            = board.D17
POLL_RATE            = 0.01   # seconds (100 Hz)

PUBLISH_ORIENTATION  = True   # roll, pitch, yaw (°) + quaternion  →  imu_roll/pitch/yaw, imu_quat_*
PUBLISH_ACCELERATION = True   # linear acceleration (m/s², gravity removed)  →  imu_linear_accel_*
PUBLISH_ANGULAR_VEL  = True   # angular velocity (°/s)  →  imu_gyro_*
AUTOCALIBRATE        = True   # define axes from startup gravity & heading;
                               # adds imu_*_cal counterparts for every active key
# ──────────────────────────────────────────────────────────────────────────────

_NEED_QUAT = PUBLISH_ORIENTATION or AUTOCALIBRATE

# ── I2C baudrate check ─────────────────────────────────────────────────────────
def check_baudrate():
    try:
        with open("/sys/class/i2c-adapter/i2c-1/of_node/clock-frequency", "rb") as f:
            current_baud = int.from_bytes(f.read(), byteorder="big")
        print(f"[IMU] I2C baudrate: {current_baud} Hz")
        if current_baud > 50000:
            print("!" * 60)
            print("WARNING: I2C baudrate is too high for BNO08x.")
            print("Set 'dtparam=i2c_arm=on,i2c_arm_baudrate=40000' in /boot/config.txt")
            print("!" * 60)
    except Exception:
        print("[IMU] Could not read I2C baudrate, skipping check.")

# ── Hardware reset ─────────────────────────────────────────────────────────────
def reset_sensor(reset_pin):
    print("[IMU] Resetting sensor...")
    reset_pin.value = False
    time.sleep(0.2)
    reset_pin.value = True
    time.sleep(0.8)

# ── Sensor init ────────────────────────────────────────────────────────────────
def init_bno(i2c, reset_pin):
    reset_sensor(reset_pin)
    try:
        bno = BNO08X_I2C(i2c, address=I2C_ADDRESS)
        bno.enable_feature(BNO_REPORT_ACCELEROMETER)
        time.sleep(0.05)
        if _NEED_QUAT:
            bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            time.sleep(0.05)
        if PUBLISH_ACCELERATION:
            bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
            time.sleep(0.05)
        if PUBLISH_ANGULAR_VEL:
            bno.enable_feature(BNO_REPORT_GYROSCOPE)
            time.sleep(0.05)
        if AUTOCALIBRATE:
            bno.enable_feature(BNO_REPORT_GRAVITY)
            time.sleep(0.05)
        print("[IMU] Sensor initialised.")
        return bno
    except Exception as e:
        print(f"[IMU] Init failed: {e}")
        return None

# ── Quaternion → Euler ─────────────────────────────────────────────────────────
def quaternion_to_euler(i, j, k, w):
    sinr_cosp = 2 * (w * i + j * k)
    cosr_cosp = 1 - 2 * (i * i + j * j)
    roll  = math.atan2(sinr_cosp, cosr_cosp)

    sinp  = 2 * (w * j - k * i)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

    siny_cosp = 2 * (w * k + i * j)
    cosy_cosp = 1 - 2 * (j * j + k * k)
    yaw_deg   = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    if yaw_deg < 0:
        yaw_deg += 360

    return math.degrees(roll), math.degrees(pitch), yaw_deg

# ── Autocalibration math ───────────────────────────────────────────────────────
def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _quat_to_matrix(i, j, k, w):
    """Quaternion (i, j, k, w) → 3×3 rotation matrix (body → world)."""
    return np.array([
        [1 - 2*(j*j + k*k),     2*(i*j - k*w),     2*(i*k + j*w)],
        [    2*(i*j + k*w), 1 - 2*(i*i + k*k),     2*(j*k - i*w)],
        [    2*(i*k - j*w),     2*(j*k + i*w), 1 - 2*(i*i + j*j)],
    ])


def _to_cal_frame(v_body, quat):
    """Rotate a body-frame vector (x, y, z) into the calibrated world frame."""
    v_world = _quat_to_matrix(*quat) @ np.array(v_body)
    return _R_cal @ v_world

# ── Autocalibration setup ──────────────────────────────────────────────────────
# Calibrated frame axes (defined once at startup):
#   Z = up (opposite to gravity)
#   X = startup forward direction (sensor +X projected onto the horizontal plane)
#   Y = startup right (X × Z)
#
# _R_cal rows are the calibrated axes in world coords → v_cal = _R_cal @ v_world
_R_cal       = None
_startup_yaw = None   # degrees; yaw offset so that startup heading = 0°


def setup_calibration(bno):
    global _R_cal, _startup_yaw
    print("[IMU] Calibrating axes — hold sensor still...")
    time.sleep(1.0)

    gravity_raw, q = None, None
    while gravity_raw is None or q is None:
        gravity_raw = bno.gravity
        q           = bno.quaternion
        time.sleep(0.05)

    Z_cal      = _normalize(-np.array(gravity_raw))          # up
    sensor_fwd = _quat_to_matrix(*q)[:, 0]                   # sensor +X in world frame
    X_cal      = _normalize(sensor_fwd - np.dot(sensor_fwd, Z_cal) * Z_cal)  # forward, horizontal
    Y_cal      = _normalize(np.cross(X_cal, Z_cal))           # right

    _R_cal = np.array([X_cal, Y_cal, Z_cal])

    _, _, _startup_yaw = quaternion_to_euler(*q)
    print(f"[IMU] Calibration done. Startup yaw: {_startup_yaw:.1f}°")
    print(f"[IMU]   Z (up):      {Z_cal.round(3)}")
    print(f"[IMU]   X (forward): {X_cal.round(3)}")
    print(f"[IMU]   Y (right):   {Y_cal.round(3)}")

# ── Main ───────────────────────────────────────────────────────────────────────
check_baudrate()

broker = TelemetryBroker()

reset_pin           = digitalio.DigitalInOut(RESET_PIN)
reset_pin.direction = digitalio.Direction.OUTPUT
reset_pin.value     = True

i2c    = busio.I2C(board.SCL, board.SDA)
sensor = init_bno(i2c, reset_pin)

if AUTOCALIBRATE:
    while sensor is None:
        print("[IMU] Waiting for sensor before calibration...")
        time.sleep(1)
        sensor = init_bno(i2c, reset_pin)
    setup_calibration(sensor)

print("[IMU] Starting orientation stream...")

while True:
    if sensor is None:
        sensor = init_bno(i2c, reset_pin)
        time.sleep(1)
        continue

    try:
        data = {}
        quat = sensor.quaternion if _NEED_QUAT else None

        # Raw accelerometer (includes gravity) — always published
        accel = sensor.acceleration
        if accel is not None:
            data["imu_accel_x"] = round(accel[0], 3)
            data["imu_accel_y"] = round(accel[1], 3)
            data["imu_accel_z"] = round(accel[2], 3)

        # Orientation: roll / pitch / yaw + quaternion
        if PUBLISH_ORIENTATION and quat is not None:
            i, j, k, w = quat
            roll, pitch, yaw = quaternion_to_euler(i, j, k, w)
            data["imu_roll"]   = round(roll,  2)
            data["imu_pitch"]  = round(pitch, 2)
            data["imu_yaw"]    = round(yaw,   2)
            data["imu_quat_i"] = round(i, 4)
            data["imu_quat_j"] = round(j, 4)
            data["imu_quat_k"] = round(k, 4)
            data["imu_quat_w"] = round(w, 4)
            if AUTOCALIBRATE:
                # Roll and pitch are gravity-referenced — identical in cal frame.
                # Yaw is offset so startup heading = 0°.
                data["imu_roll_cal"]  = round(roll,  2)
                data["imu_pitch_cal"] = round(pitch, 2)
                data["imu_yaw_cal"]   = round((yaw - _startup_yaw) % 360, 2)

        # Linear acceleration (gravity removed)
        if PUBLISH_ACCELERATION:
            la = sensor.linear_acceleration
            if la is not None:
                data["imu_linear_accel_x"] = round(la[0], 3)
                data["imu_linear_accel_y"] = round(la[1], 3)
                data["imu_linear_accel_z"] = round(la[2], 3)
                if AUTOCALIBRATE and quat is not None:
                    cx, cy, cz = _to_cal_frame(la, quat)
                    data["imu_linear_accel_cal_x"] = round(float(cx), 3)
                    data["imu_linear_accel_cal_y"] = round(float(cy), 3)
                    data["imu_linear_accel_cal_z"] = round(float(cz), 3)

        # Angular velocity
        if PUBLISH_ANGULAR_VEL:
            gyr = sensor.gyro
            if gyr is not None:
                data["imu_gyro_x"] = round(math.degrees(gyr[0]), 2)
                data["imu_gyro_y"] = round(math.degrees(gyr[1]), 2)
                data["imu_gyro_z"] = round(math.degrees(gyr[2]), 2)
                if AUTOCALIBRATE and quat is not None:
                    cx, cy, cz = _to_cal_frame(gyr, quat)
                    data["imu_gyro_cal_x"] = round(math.degrees(float(cx)), 2)
                    data["imu_gyro_cal_y"] = round(math.degrees(float(cy)), 2)
                    data["imu_gyro_cal_z"] = round(math.degrees(float(cz)), 2)

        if data:
            broker.setmulti(data)

        time.sleep(POLL_RATE)

    except KeyboardInterrupt:
        print("\n[IMU] Stopped.")
        broker.close()
        break
    except Exception as e:
        print(f"[IMU] {type(e).__name__}: {e} — reinitialising...")
        sensor = None
