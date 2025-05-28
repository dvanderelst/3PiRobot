from pololu_3pi_2040_robot import robot
import math
import settings

class Motors:
    def __init__(self):
        counts_per_rev = 909.72
        wheel_diameter_mm = 31
        self.verbose = settings.verbose
        self.max_mps = 0.4 #This comes from the specs. Probably conservative.
        self.counts_per_rev = counts_per_rev
        self.wheel_diameter_m = wheel_diameter_mm / 1000  # convert to meters
        self.wheel_circumference_m = math.pi * self.wheel_diameter_m
        self.meters_per_count = self.wheel_circumference_m / self.counts_per_rev
        self.counts_per_meter = self.counts_per_rev / self.wheel_circumference_m
        self.max_cps = self.mps2cps(self.max_mps)
        self.motors = robot.Motors()
    
    def check_cps(self, cps):
        if cps > self.max_cps: cps = self.max_cps
        if cps < -self.max_cps: cps = -self.max_cps
        return cps
    
    def check_mps(self, mps):
        if mps > self.max_mps: mps = self.max_mps
        if mps < -self.max_mps: mps = -self.max_mps
        return mps
    
    def cps2mps(self, cps):
        cps = self.check_cps(cps)
        return cps * self.meters_per_count

    def mps2cps(self, mps):
        mps = self.check_mps(mps)
        return mps * self.counts_per_meter
    
    def set_speeds(self, left, right=None):
        if right is None: right = left * 1.0
        left_speed = self.check_mps(left)
        right_speed = self.check_mps(right)
        left_cps = self.mps2cps(left_speed)
        right_cps = self.mps2cps(right_speed)
        self.motors.set_speeds(left_cps, right_cps)
        if self.verbose: print(f'[MTR] L {left_speed}, R {right_speed}')
    
    def stop(self):
        self.set_speeds(0)
