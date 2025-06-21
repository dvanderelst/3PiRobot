# drive_base.py  – high-level motion helper for the Pololu 3pi+ 2040
from pololu_3pi_2040_robot import robot
import math, time, settings


class DriveBase:
    """
    High-level motion helper that works in physical units (m, m/s, deg).
    Internally it converts to PWM for the motors and *checks* progress
    with the wheel encoders so moves stop exactly at the requested distance
    or angle.
    """

    # ─────────── robot-specific constants  (edit for your edition) ────────────
    GEAR_RATIO        = 75          # 30:1 Standard, 75:1 Turtle, 15:1 Hyper
    COUNTS_PER_REV    = 12          # encoder counts per motor-shaft rev
    WHEEL_DIAMETER_MM = 32          # measure yours!
    WHEEL_BASE_MM     = 85          # centre-to-centre of wheels
    MAX_WHEEL_MPS     = 0.40        # top speed you measured
    MAX_PWM           = 6000        # from motors.py
    MIN_PWM           = 600         # ≈ first PWM that makes wheels turn

    # ───────────────────────── derived constants ─────────────────────────────
    COUNTS_PER_WHEEL_REV = COUNTS_PER_REV * GEAR_RATIO
    WHEEL_CIRCUMFERENCE  = math.pi * WHEEL_DIAMETER_MM / 1000          # m
    COUNTS_PER_METER     = COUNTS_PER_WHEEL_REV / WHEEL_CIRCUMFERENCE
    WHEEL_BASE_M         = WHEEL_BASE_MM / 1000

    def __init__(self, verbose=True):
        self.verbose  = verbose and getattr(settings, "verbose", True)
        self.motors   = robot.Motors()
        self.encoders = robot.Encoders()

    # ───────────────────────── helpers ───────────────────────────────────────
    @staticmethod
    def _sgn(x): return 1 if x >= 0 else -1

    def _mps_to_pwm(self, mps: float) -> int:
        """Convert wheel speed (m/s) → PWM (−6000…+6000) with a dead-band."""
        mps = max(-self.MAX_WHEEL_MPS, min(self.MAX_WHEEL_MPS, mps))
        pwm = mps / self.MAX_WHEEL_MPS * self.MAX_PWM
        if 0 < abs(pwm) < self.MIN_PWM:
            pwm = math.copysign(self.MIN_PWM, pwm)
        return int(pwm)

    def _set_pwm(self, left_mps: float, right_mps: float) -> None:
        l_cmd = self._mps_to_pwm(left_mps)
        r_cmd = self._mps_to_pwm(right_mps)
        self.motors.set_speeds(l_cmd, r_cmd)
        if self.verbose:
            print(f"[PWM] L {left_mps:+.3f} m/s → {l_cmd:+4d} | R {right_mps:+.3f} m/s → {r_cmd:+4d}")

    # ───────────────────── high-level velocity command ───────────────────────
    def set_kinematics(self, lin_mps: float = 0.0, rot_dps: float = 0.0):
        """
        lin_mps : forward (+) or backward (−) linear speed in m/s
        rot_dps : clockwise (+) or CCW (−) angular speed in deg/s
        """
        ω = -math.radians(rot_dps)                               # +CW
        v_l = lin_mps - (self.WHEEL_BASE_M / 2) * ω
        v_r = lin_mps + (self.WHEEL_BASE_M / 2) * ω
        self._set_pwm(v_l, v_r)

    # ───────────────────── drive exact distance (encoder) ────────────────────
    def drive_distance(self, meters: float, speed: float = 0.20):
        """
        Drive `meters` forward (+) or backward (−) at up to `speed` (m/s),
        stopping when the encoders say we've arrived.
        """
        speed   = min(abs(speed), self.MAX_WHEEL_MPS * 0.8)  # margin
        counts  = abs(meters) * self.COUNTS_PER_METER
        start_L, start_R = self.encoders.get_counts()
        direction = self._sgn(meters)

        self._set_pwm(direction * speed, direction * speed)

        while True:
            cur_L, cur_R = self.encoders.get_counts()
            moved = ((cur_L - start_L) + (cur_R - start_R)) / 2
            if moved >= counts:
                break
            time.sleep(0.005)

        self.stop()

    # ───────────────────── rotate exact angle (encoder) ──────────────────────
    def turn_angle(self, angle_deg: float, wheel_mps: float = 0.10):
        """
        Rotate by `angle_deg` (CW +, CCW −) using encoders for accuracy.
        """
        wheel_mps = min(abs(wheel_mps), self.MAX_WHEEL_MPS * 0.6)
        arc_m     = math.pi * self.WHEEL_BASE_M * abs(angle_deg) / 360
        counts    = arc_m * self.COUNTS_PER_METER
        start_L, start_R = self.encoders.get_counts()
        sign = self._sgn(angle_deg)

        # wheels spin opposite ways for on-the-spot turn
        self._set_pwm(-sign * wheel_mps, sign * wheel_mps)

        while True:
            cur_L, cur_R = self.encoders.get_counts()
            if (abs(cur_L - start_L) >= counts and
                abs(cur_R - start_R) >= counts):
                break
            time.sleep(0.005)

        self.stop()

    # ─────────────────────────── util ────────────────────────────────────────
    def stop(self):
        self.motors.set_speeds(0, 0)
        if self.verbose:
            print("[STOP]")

# ─────────────────────────── demo ────────────────────────────────────────────
if __name__ == "__main__":
    settings.verbose = True
    bot = DriveBase()

    print("Rotate 90° clockwise …")
    bot.turn_angle(90)

    #time.sleep(1)

    #bot.drive_distance(0.30)

    #print("Rotate 45° CCW …")
    #bot.turn_angle(-45)

    #bot.stop()
    #print("Done.")
