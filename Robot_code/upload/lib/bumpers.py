from pololu_3pi_2040_robot import ir_sensors
import time


class Bumpers:
    def __init__(self, leds=None):
        self._bump = ir_sensors.BumpSensors()
        self._bump.reset_calibration()
        self._bump.calibrate()
        self.leds = leds

    def read(self,update_leds=True):
        """Read the bump sensors and return a (left, right) tuple of booleans."""
        self._bump.read()
        bump_left = self.left()
        bump_right = self.right()
        if update_leds and self.leds is not None:
            self.leds.set(5, 'red') if bump_left else self.leds.set(5, 'black')
            self.leds.set(3, 'red') if bump_right else self.leds.set(3, 'black')
        return bump_left, bump_right

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

