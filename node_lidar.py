from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from lidar_utils.lidar_read import read_lidar_data, SensorUnavailableError
from lidar_utils import lidar_sim
from lidar_utils.lidar_analysis import simple_corners
import json
import math

SIM_REPLACE = True   # Use simulation if sensor is not found
VISUALISE = True     # Show live polar plot of lidar data

mb = TelemetryBroker()

data_dict = {}
angle_dict = {}

if VISUALISE:
    from lidar_utils.lidar_vis import LiveVisualiser
    vis = LiveVisualiser()
else:
    vis = None

_prev_angle = None


def on_measurement(angle, distance, quality):
    global _prev_angle
    angle_dict[angle] = distance
    data_dict["lidar"] = json.dumps(angle_dict)
    mb.setmulti(data_dict)

    if _prev_angle is not None and angle < _prev_angle:
        # Full scan completed — run corner detection
        sorted_angles = sorted(angle_dict.keys())
        points = [
            (
                (angle_dict[a] / 1000) * math.cos(math.radians(a)),
                (angle_dict[a] / 1000) * math.sin(math.radians(a))
            )
            for a in sorted_angles
        ]
        corner_xy = simple_corners(points)
        corners = [
            (math.degrees(math.atan2(y, x)) % 360, math.hypot(x, y) * 1000)
            for x, y in corner_xy
        ]
        data_dict["lidar_corners"] = json.dumps(corners)
        mb.setmulti(data_dict)

        if vis is not None:
            vis.update(angle_dict, corners)

    _prev_angle = angle


if __name__ == "__main__":
    try:
        read_lidar_data(on_measurement)
    except SensorUnavailableError:
        if SIM_REPLACE:
            print("Falling back to simulated sensor data.")
            lidar_sim.read_lidar_data(on_measurement)
        else:
            raise
