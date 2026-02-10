from pololu_3pi_2040_robot import robot
import math, time, settings


class Motors:
    gear_ratio       = 75
    counts_per_rev   = 12
    max_wheel_mps    = 0.40
    max_pwm          = 6000
    min_pwm          = 600

    def __init__(self):
        self.verbose           = settings.verbose
        self.motors            = robot.Motors()
        self.encoders          = robot.Encoders()
        self.wheel_base_mm     = 84.5
        self.wheel_diameter_mm = 32
        self._update_derived_constants()
        self._last_left_pwm = 0  # Track last set PWM values
        self._last_right_pwm = 0

    def _update_derived_constants(self):
        self.wheel_base_m = self.wheel_base_mm / 1000
        self.wheel_circumference = math.pi * self.wheel_diameter_mm / 1000
        self.counts_per_wheel_rev = self.counts_per_rev * self.gear_ratio
        self.counts_per_meter = self.counts_per_wheel_rev / self.wheel_circumference

    def set_wheel_base(self, mm):
        self.wheel_base_mm = mm
        self._update_derived_constants()

    def set_wheel_diameter(self, mm):
        self.wheel_diameter_mm = mm
        self._update_derived_constants()

    @staticmethod
    def _sgn(x): return 1 if x >= 0 else -1

    def _mps_to_pwm(self, mps: float) -> int:
        mps = max(-self.max_wheel_mps, min(self.max_wheel_mps, mps))
        pwm = mps / self.max_wheel_mps * self.max_pwm
        if 0 < abs(pwm) < self.min_pwm: pwm = math.copysign(self.min_pwm, pwm)
        return int(pwm)

    def _set_pwm(self, left_mps: float, right_mps: float) -> None:
        l_cmd = self._mps_to_pwm(left_mps)
        r_cmd = self._mps_to_pwm(right_mps)
        self.motors.set_speeds(l_cmd, r_cmd)
        # Track the last set PWM values
        self._last_left_pwm = l_cmd
        self._last_right_pwm = r_cmd
        if self.verbose: print(f"[PWM] L {left_mps:+.3f} m/s → {l_cmd:+4d} | R {right_mps:+.3f} m/s → {r_cmd:+4d}")

    def set_kinematics(self, lin_mps: float = 0.0, rot_dps: float = 0.0):
        omega = -math.radians(rot_dps)  # +CW
        v_l = lin_mps - (self.wheel_base_m / 2) * omega
        v_r = lin_mps + (self.wheel_base_m / 2) * omega
        self._set_pwm(v_l, v_r)

    def drive_distance(self, meters: float, speed: float = 0.20):
        speed = min(abs(speed), self.max_wheel_mps * 0.8)
        counts = abs(meters) * self.counts_per_meter
        start_l, start_r = self.encoders.get_counts()
        direction = self._sgn(meters)

        self._set_pwm(direction * speed, direction * speed)

        while True:
            cur_l, cur_r = self.encoders.get_counts()
            moved = ((cur_l - start_l) + (cur_r - start_r)) / 2
            if moved >= counts:
                break
            time.sleep(0.001)

        self.stop()

    def turn_angle(self, angle_deg: float, rot_speed_dps: float = 90.0):
        omega_rad = math.radians(rot_speed_dps)
        wheel_mps = abs(omega_rad * self.wheel_base_m / 2)
        wheel_mps = min(wheel_mps, self.max_wheel_mps * 0.6)

        arc_m = math.pi * self.wheel_base_m * abs(angle_deg) / 360
        counts = arc_m * self.counts_per_meter

        start_l, start_r = self.encoders.get_counts()
        sign = self._sgn(angle_deg)

        self._set_pwm(sign * wheel_mps, -sign * wheel_mps)

        while True:
            cur_l, cur_r = self.encoders.get_counts()
            if (abs(cur_l - start_l) >= counts or abs(cur_r - start_r) >= counts):
                break
            time.sleep(0.001)

        self.stop()

    def stop(self):
        self.motors.set_speeds(0, 0)
        # Update tracked PWM values to reflect stopped state
        self._last_left_pwm = 0
        self._last_right_pwm = 0
        if self.verbose: print("[STOP]")

    def is_moving(self):
        """Check if robot is moving based on tracked PWM values"""
        return self._last_left_pwm != 0 or self._last_right_pwm != 0
