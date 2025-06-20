from pololu_3pi_2040_robot import robot
import math
import settings

class Motors:
    def __init__(self):
        self.verbose = settings.verbose
        self.wheel_diameter_mm = 31
        self.wheel_base_mm = 170
        self.max_mps = 0.4 # m/s, This comes from the specs. Probably conservative.
        self.counts_per_rev = 909.72
        
        self.wheel_base_m = self.wheel_base_mm / 1000
        self.wheel_diameter_m = self.wheel_diameter_mm / 1000  # convert to meters
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
    
    def set_speeds(self, left, right=None): #m/sec
        if right is None: right = left * 1.0
        left_speed = self.check_mps(left)
        right_speed = self.check_mps(right)
        left_cps = self.mps2cps(left_speed)
        right_cps = self.mps2cps(right_speed)
        self.motors.set_speeds(left_cps, right_cps)
        if self.verbose: print(f'[MTR] LFT {left_speed}, RGT {right_speed}')
    
    def set_kinematics(self, lin_speed=0, rot_speed=0): #m/s and deg/sec
        omega =  - rot_speed * 0.0174533 # This makes positive = right rotation
        # Standard differential drive inverse kinematics
        left = lin_speed - (self.wheel_base_m / 2) * omega
        right = lin_speed + (self.wheel_base_m / 2) * omega
        # Clamp wheel speeds to max_mps
        left = self.check_mps(left)
        right = self.check_mps(right)
        # Compute actual robot linear and angular speeds
        v = (left + right) / 2
        omega_deg = (right - left) / self.wheel_base_m / 0.0174533  # rad/s to deg/s
        self.set_speeds(left, right)
        if self.verbose: print(f'[MTR] LIN {v}, ROT {omega_deg}')
        return left, right, v, omega_deg

    def drive_distance(self, meters, speed=None):
        speed = self.check_mps(speed or 0.2)
        duration = abs(meters / speed)
        direction = 1 if meters >= 0 else -1
        self.set_speeds(direction * speed)
        time.sleep(duration)
        self.stop()

    def turn_angle(self, angle_deg, rot_speed=None):
        rot_speed = rot_speed or (90/5)  # deg/sec
        direction = 1 if angle_deg >= 0 else -1
        duration = abs(angle_deg / rot_speed)
        self.kinematics(0, direction * rot_speed)
        time.sleep(duration)
        self.stop()


    def stop(self):
        self.set_speeds(0)
        



