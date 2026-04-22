from pololu_3pi_2040_robot import ir_sensors
import time


class Bumpers:
    def __init__(self, leds=None):
        self._bump = ir_sensors.BumpSensors()
        self.leds = leds
        self.left_threshold = 250
        self.right_threshold = 250
        self._last_left = None
        self._last_right = None

    def read(self, update_leds=True):
        """Read the bump sensors and return a (left, right) tuple of booleans."""
        values = self._bump.read()
        bump_left = values[0] > self.left_threshold
        bump_right = values[1] > self.right_threshold
        if update_leds and self.leds is not None:
            if bump_left != self._last_left:
                self.leds.set(5, 'red' if bump_left else 'black')
            if bump_right != self._last_right:
                self.leds.set(3, 'red' if bump_right else 'black')
        self._last_left = bump_left
        self._last_right = bump_right
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

