from pololu_3pi_2040_robot import robot
import math

class Controller:
    def __init__(self):
        """Initialize with encoder counts per revolution and wheel diameter in mm."""
        counts_per_rev = 909.72
        wheel_diameter_mm = 31
        self.max_mps = 0.4 #This comes from the specs. Probably conservative.
        self.counts_per_rev = counts_per_rev
        self.wheel_diameter_m = wheel_diameter_mm / 1000  # convert to meters
        self.wheel_circumference_m = math.pi * self.wheel_diameter_m
        self.meters_per_count = self.wheel_circumference_m / self.counts_per_rev
        self.counts_per_meter = self.counts_per_rev / self.wheel_circumference_m
        self.max_cps = self.mps2cps(self.max_mps)
        self.motors = robot.Motors()
    
    def check_cps(self, cps):
        #if cps < 0: cps = 0
        if cps > self.max_cps: cps = self.max_cps
        if cps < -self.max_cps: cps = -self.max_cps
        return cps
    
    def check_mps(self, mps):
        #if mps < 0: mps = 0
        if mps > self.max_mps: mps = self.max_mps
        if mps < -self.max_mps: mps = -self.max_mps
        ret