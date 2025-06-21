from pololu_3pi_2040_robot import robot
import math
import time
import settings

class Motors:
    def __init__(self):
        self.verbose = settings.verbose
        self.wheel_diameter_mm = 31
        self.wheel_base_mm = 85
        
        self.max_mps = 0.4       # maximum linear wheel speed in m/s
        self.MAX_PWM = 6000      # maximum PWM value (from motors.py)
        self.MIN_PWM = 600       # minimum PWM to overcome static friction
        
        self.wheel_base_m = self.wheel_base_mm / 1000
        self.wheel_diameter_m = self.wheel_diameter_mm / 1000  # convert to meters
        self.wheel_circumference_m = math.pi * self.wheel_diameter_m
        
        self.motors = robot.Motors()
    
    def check_cps(self, cps):
        if cps > self.max_cps: cps = self.max_cps
        if cps < -self.max_cps: cps = -self.max_cps
        return cps
    
    def check_mps(self, mps):
        if mps > self.max_mps: mps = self.max_mps
        if mps < -self.max_mps: mps = -self.max_mps
        return mps
    
    def mps2pwm(self, mps):
        mps = self.check_mps(mps)  # clamp to safe range
        pwm = mps/self.max_mps * self.MAX_PWM
        if 0 < abs(pwm) < self.MIN_PWM: pwm = math.copysign(self.MIN_PWM, pwm)
        return int(pwm)
    
    def set_speeds(self, left, right=None): #m/sec
        if right is None: right = left * 1.0
        left_speed = self.check_mps(left)
        right_speed = self.check_mps(right)
        left_pwm = self.mps2pwm(left_speed)
        right_pwm = self.mps2pwm(right_speed)
        self.motors.set_speeds(left_pwm, right_pwm)
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

    def drive_distance(self, meters, lin_speed=None):
        lin_speed = self.check_mps(lin_speed or 0.2)
        duration = abs(meters / lin_speed)
        direction = 1 if meters >= 0 else -1
        self.set_speeds(direction * lin_speed)
        time.sleep(duration)
        self.stop()

    def turn_angle(self, angle_deg, rot_speed=None):
        rot_speed = rot_speed or 90  # deg/sec
        direction = 1 if angle_deg >= 0 else -1
        duration = abs(angle_deg / rot_speed)
        self.set_kinematics(0, direction * rot_speed)
        time.sleep(duration)
        self.stop()


    def stop(self):
        self.set_speeds(0)
        


d = Motors()
d.turn_angle(45)

