try:
    import serial
    _serial_import_error = None
except ImportError as e:
    print(e, "\nTrying pyserial...")
    try:
        import pyserial as serial
        _serial_import_error = None
    except ImportError as e:
        print(e, "\nBoth serial and pyserial failed")
        serial = None
        _serial_import_error = e

PORT = ''
BAUD = 460800

class SensorUnavailableError(Exception):
    pass


def read_lidar_data(on_update):
    """
    Reads data from the RPLidar C1 sensor continuously.
    Calls on_update(angle, distance, quality) for each valid measurement.
    Stops on KeyboardInterrupt.
    Raises SensorUnavailableError if the serial module is missing or the
    sensor cannot be opened.
    """
    if serial is None:
        print("Serial module not available")
        raise SensorUnavailableError(
            f"serial module not available: {_serial_import_error}"
        )

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"Sensor not found on {PORT}: {e}")
        raise SensorUnavailableError(f"Sensor not found on {PORT}: {e}") from e

    ser.flushInput()  # Clear old buffer

    print("Starting scan...")
    ser.write(b'\xa5\x20')  # Send scan command

    # Skip the 7-byte response header (A5 5A ...)
    descriptor = ser.read(7)
    print(f"Lidar response: {descriptor.hex()}")

    print("Reading measurements (Ctrl+C to stop)...")
    try:
        while True:
            # One standard data point is 5 bytes long
            raw_data = ser.read(5)
            if len(raw_data) < 5:
                continue

            # --- PARSE DATA ---
            # Byte 0: check bits and quality
            quality = raw_data[0] >> 2

            # Bytes 1 & 2: angle (transmitted in 1/64 degrees)
            angle_raw = (raw_data[1] >> 1) | (raw_data[2] << 7)
            angle = int(angle_raw / 64.0)

            # Bytes 3 & 4: distance (millimeters)
            # This is the critical spot for the 65000 issue!
            dist_raw = raw_data[3] | (raw_data[4] << 8)
            distance = int(dist_raw / 4.0)  # C1 uses quarter-millimeters

            # Filter: only pass plausible values (max 12 meters)
            if distance > 0 and distance < 12000:
                on_update(angle, distance, quality)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        ser.write(b'\xa5\x25')  # Stop command
        ser.close()
