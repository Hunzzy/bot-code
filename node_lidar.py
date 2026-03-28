from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from lidar_utils import lidar_read_usb, lidar_read_uart, lidar_sim
import json
import queue

SIM_REPLACE = True  # Use simulation if sensor is not found
BATCH_SIZE  = 360   # Publish to broker every N measurements

mb = TelemetryBroker()

angle_dict   = {}
_batch_count = 0


def on_measurement(angle, distance, quality):
    global _batch_count
    angle_dict[int(round(angle))] = distance
    _batch_count += 1
    if _batch_count >= BATCH_SIZE:
        mb.set("lidar", json.dumps(angle_dict))
        _batch_count = 0


def on_scan(batch):
    """Batch callback for simulation: receives a full {angle: dist_mm} dict at once."""
    angle_dict.update(batch)
    mb.set("lidar", json.dumps(angle_dict))


if __name__ == "__main__":
    raw_queue = queue.Queue(maxsize=36000)  # ~10 full scans of headroom

    # Try USB first, then UART0, then simulation.
    for _reader in (lidar_read_usb, lidar_read_uart):
        try:
            producer = _reader.start_producer(raw_queue)
            print(f"Sensor opened on {_reader.PORT}")
            print("Reading measurements (Ctrl+C to stop)...")
            try:
                while True:
                    result = _reader.parse_packet(raw_queue.get())
                    if result:
                        on_measurement(*result)
            except KeyboardInterrupt:
                print("\nStopping...")
                producer.stop()
            break
        except _reader.SensorUnavailableError as e:
            print(f"[{_reader.PORT}] not available: {e}")
    else:
        if SIM_REPLACE:
            print("Falling back to simulated sensor data.")

            def _on_sim_ready(px, py, angle_f):
                mb.set("sim_heading", str(round(angle_f, 1)))

            def _on_sim_heading(heading_deg):
                mb.set("sim_heading", str(round(heading_deg, 1)))

            lidar_sim.read_lidar_data(on_measurement, on_ready=_on_sim_ready,
                                      on_heading=_on_sim_heading,
                                      on_scan=on_scan)
            # Per-ray fallback (realistic drip-feed, matches real hardware):
            # lidar_sim.read_lidar_data(on_measurement, on_ready=_on_sim_ready,
            #                           on_heading=_on_sim_heading)
        else:
            raise lidar_read_usb.SensorUnavailableError("No sensor found on any port.")
