from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from lidar_utils.lidar_read import read_lidar_data, SensorUnavailableError
from lidar_utils import lidar_sim
import json

SIM_REPLACE = True  # Use simulation if sensor is not found
BATCH_SIZE = 360    # Publish to broker every N measurements

mb = TelemetryBroker()

angle_dict = {}
_batch_count = 0


def on_measurement(angle, distance, quality):
    global _batch_count
    angle_dict[angle] = distance
    _batch_count += 1
    if _batch_count >= BATCH_SIZE:
        mb.set("lidar", json.dumps(angle_dict))
        _batch_count = 0


if __name__ == "__main__":
    try:
        read_lidar_data(on_measurement)
    except SensorUnavailableError:
        if SIM_REPLACE:
            print("Falling back to simulated sensor data.")
            lidar_sim.read_lidar_data(on_measurement)
        else:
            raise
