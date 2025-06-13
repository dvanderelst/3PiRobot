from pololu_3pi_2040_robot import robot
import time
import settings

class bumpers:
    def __init__(self):
        self.verbose = settings.verbose
        self.bump_sensors = robot.BumpSensors()
        self.bump_sensors.calibrate()
        time.sleep_ms(1000)
        
    def read(self):
        left = self.bump_sensors.left_is_pressed()
        right = self.bump_sensors.right_is_pressed()
        return [left, right]
        