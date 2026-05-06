from my_pololu_lib import robot
import math, time, settings

_INTER_PHASE_DELAY_MS = 100  # pause between turn and drive within a step
_PHASE_TIMEOUT_MS     = 5000 # max ms for a single turn or drive phase before force-stop


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

        # Cooperative motion state — main loop must call tick() each iteration
        self._active = None           # current phase dict, or None when idle
        self._plan = []               # queued phases pending
        self._pending_cmd_id = None   # cmd to reply 'done' to when plan completes

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
        if self.verbose: print(f"[PWM] L {left_mps:+.3f} m/s → {l_cmd:+4d} | R {right_mps:+.3f} m/s → {r_cmd:+4d}")

    def set_kinematics(self, lin_mps: float = 0.0, rot_dps: float = 0.0):
        omega = math.radians(rot_dps)  # +CCW (left turn) — see rotation_conventions on Notion
        v_l = lin_mps - (self.wheel_base_m / 2) * omega
        v_r = lin_mps + (self.wheel_base_m / 2) * omega
        self._set_pwm(v_l, v_r)

    def stop(self):
        self.motors.set_speeds(0, 0)
        if self.verbose: print("[STOP]")

    # ─── Cooperative motion state machine ───
    def is_busy(self):
        return self._active is not None or bool(self._plan)

    def begin_step(self, distance=0, angle=0, linear_speed=0, rotation_speed=0, cmd_id=None):
        """Queue a turn-then-drive step. Returns False if already busy."""
        if self.is_busy(): return False
        if abs(angle) > 0 and rotation_speed == 0: rotation_speed = 90
        if abs(distance) > 0 and linear_speed == 0: linear_speed = 0.1
        plan = []
        if abs(angle) > 0:
            plan.append(('turn', angle, rotation_speed))
        if abs(angle) > 0 and abs(distance) > 0:
            plan.append(('wait', _INTER_PHASE_DELAY_MS, None))
        if abs(distance) > 0:
            plan.append(('drive', distance, linear_speed))
        self._plan = plan
        self._pending_cmd_id = cmd_id
        return True

    def abort(self, reason='aborted'):
        """Stop motors and clear any queued motion. Returns the pending cmd_id, if any."""
        had_pending = self._pending_cmd_id is not None
        self.stop()
        self._active = None
        self._plan = []
        cmd_id = self._pending_cmd_id
        self._pending_cmd_id = None
        return cmd_id if had_pending else None

    def _start_phase(self, phase):
        kind, a, b = phase
        if kind == 'turn':
            angle_deg, rot_speed_dps = a, b
            omega_rad = math.radians(rot_speed_dps)
            wheel_mps = min(abs(omega_rad * self.wheel_base_m / 2), self.max_wheel_mps * 0.6)
            arc_m = math.pi * self.wheel_base_m * abs(angle_deg) / 360
            counts = arc_m * self.counts_per_meter
            start_l, start_r = self.encoders.get_counts()
            sign = self._sgn(angle_deg)
            # +angle_deg = CCW (left turn): right wheel forward, left wheel backward.
            self._set_pwm(-sign * wheel_mps, sign * wheel_mps)
            self._active = {'kind': 'turn', 'start_l': start_l, 'start_r': start_r, 'counts': counts, 'start_ms': time.ticks_ms()}
        elif kind == 'drive':
            meters, speed = a, b
            speed = min(abs(speed), self.max_wheel_mps * 0.8)
            counts = abs(meters) * self.counts_per_meter
            start_l, start_r = self.encoders.get_counts()
            direction = self._sgn(meters)
            self._set_pwm(direction * speed, direction * speed)
            self._active = {'kind': 'drive', 'start_l': start_l, 'start_r': start_r, 'counts': counts, 'start_ms': time.ticks_ms()}
        elif kind == 'wait':
            duration_ms = a
            self._active = {'kind': 'wait', 'end_ms': time.ticks_add(time.ticks_ms(), duration_ms)}

    def tick(self):
        """Advance the state machine. Returns the pending cmd_id when the plan just finished, else None."""
        if self._active is None:
            if not self._plan: return None
            self._start_phase(self._plan.pop(0))
            return None

        a = self._active
        kind = a['kind']
        done = False
        if kind == 'turn':
            cur_l, cur_r = self.encoders.get_counts()
            if abs(cur_l - a['start_l']) >= a['counts'] or abs(cur_r - a['start_r']) >= a['counts']:
                self.stop(); done = True
            elif time.ticks_diff(time.ticks_ms(), a['start_ms']) > _PHASE_TIMEOUT_MS:
                print('[Motors] Turn phase timed out — stopping')
                self.stop(); done = True
        elif kind == 'drive':
            cur_l, cur_r = self.encoders.get_counts()
            moved = ((cur_l - a['start_l']) + (cur_r - a['start_r'])) / 2
            if moved >= a['counts']:
                self.stop(); done = True
            elif time.ticks_diff(time.ticks_ms(), a['start_ms']) > _PHASE_TIMEOUT_MS:
                print('[Motors] Drive phase timed out — stopping')
                self.stop(); done = True
        elif kind == 'wait':
            if time.ticks_diff(time.ticks_ms(), a['end_ms']) >= 0:
                done = True

        if done:
            self._active = None
            if not self._plan:
                cmd_id = self._pending_cmd_id
                self._pending_cmd_id = None
                return cmd_id
        return None

    # ─── Blocking convenience wrappers (standalone use / motors_test) ───
    def drive_distance(self, meters: float, speed: float = 0.20):
        self.abort()
        self._plan = [('drive', meters, speed)]
        while self.is_busy():
            self.tick()
            time.sleep(0.001)

    def turn_angle(self, angle_deg: float, rot_speed_dps: float = 90.0):
        self.abort()
        self._plan = [('turn', angle_deg, rot_speed_dps)]
        while self.is_busy():
            self.tick()
            time.sleep(0.001)
