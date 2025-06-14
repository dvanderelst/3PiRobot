from pololu_3pi_2040_robot import ir_sensors
import time

class Bumpers:
    def __init__(self):
        self._bump = ir_sensors.BumpSensors()
        self._bump.reset_calibration()
        self._bump.calibrate()

    def read(self):
        """Read the bump sensors and return a (left, right) tuple of booleans."""
        self._bump.read()
        return (self.left(), self.right())

    def left(self):
        """Return True if the left bump sensor is pressed."""
        return self._bump.left_is_pressed()

    def right(self):
        """Return True if the right bump sensor is pressed."""
        return self._bump.right_is_pressed()

    def left_changed(self):
        """Return True if the left bump state has changed since last read."""
        return self._bump.left_changed()

    def right_changed(self):
        """Return True if the right bump state has changed since last read."""
        return self._bump.right_changed()


if __name__ == "__main__":
    bump = Bumpers()
    print("Monitoring bump sensors (Press Ctrl+C to stop)...")
    try:
        while True:
            left, right = bump.read()
            print(f"Left: {left}, Right: {right}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped.")
