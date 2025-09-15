import screen
import leds
import sonar
import time
import motors
import wifi
import settings
import bumpers
import beeps
import utils

# ─────────────────────────────────────────────────────────────
# Ticks helpers: wrap once and reuse everywhere
# ─────────────────────────────────────────────────────────────
ticks_ms   = time.ticks_ms
ticks_add  = time.ticks_add
ticks_diff = time.ticks_diff


def set_free_run(val_ms, state):
    """
    Enable/disable the free-run pulser and (re)grid its next deadline.
    val_ms: period in milliseconds (0 disables)
    state:  dict holding scheduler state (period, next_due, last_mark, display)
    """
    val = max(0, int(val_ms))
    state['period'] = val
    if val > 0:
        now = ticks_ms()
        state['last_mark'] = now               # last actual pulse time
        state['next_due'] = ticks_add(now, val)  # schedule first deadline from now
        state['display'].write(4, f'Free: {val}ms')
    else:
        state['next_due'] = None
        state['display'].write(4, 'Free: off')

def main(selected_ssid=None):
    beeper = beeps.Beeper()
    # ───────────────────── Initialization ─────────────────────
    beeper.play('startup_proc')
    print('[Main] Initializing systems...')
    verbose = settings.verbose

    heartbeat_interval = 1000  # ms
    minimum_free_range_period = settings.minimum_free_range_period
    measure_guard_ms = settings.measure_guard_ms


    led = leds.LEDs()
    display = screen.Screen()
    bump = bumpers.Bumpers(leds=led)
    drive = motors.Motors()
    pulse_avg = utils.RollingAverage(10)  # tracks achieved free-run intervals

    display.clear()
    display.write(0, 'Systems up')
    led.set_all('off')

    # ───────────────────── Wi-Fi Setup ─────────────────────
    bridge, ip, ssid = wifi.setup_wifi(ssids=selected_ssid)
    display.clear()
    display.write(0, 'Connected')
    display.write(1, ssid or '')
    display.write(2, ip or '')

    # ───────────────────── Main Loop ─────────────────────
    print('[Main] Entering main loop...')
    current_led_color = 'blue'
    led.set(2, current_led_color)
    last_heartbeat = ticks_ms()

    command_queue = []

    display.write(0, 'Ready')

    # ── Free-run pulser state ──
    state = {
        'period': 0,          # ms; 0 = disabled
        'next_due': None,     # absolute deadline (ticks_ms)
        'last_mark': ticks_ms(),  # last actual pulse timestamp (ticks_ms)
        'display': display,
    }
    set_free_run(0, state)  # start disabled
    #Set color toggling
    led.set_toggle_colors(0, ['black', 'green', 'red', 'blue'])
    led.set_toggle_colors(2, leds.COLORMAPS['blue_black'])
    beeper.play('main_loop')

    # Initialize sonar. Wait until here to avoid clash with buzzer
    snr = sonar.Sonar()

    while True:
        # Cache "now" once per loop for consistent timing decisions
        now = ticks_ms()

        # ── Wi-Fi Commands ──
        new_cmds = bridge.read_messages()
        if new_cmds: command_queue.extend(new_cmds)

        # ── Safety: Bumpers ──
        bump_left, bump_right = bump.read()
        if bump_left or bump_right: drive.stop()

        # ── Command Processing ──
        if command_queue:
            cmds = command_queue
            command_queue = []
            if verbose: print(f"[Main] Received: {cmds}")

            for cmd in cmds:
                action = cmd.get('action')
                # Drive continuously (teleop-style)
                if action == 'kinematics':
                    display.write(0, 'kinematics')
                    rotation_speed = cmd.get('rotation_speed')
                    linear_speed = cmd.get('linear_speed')
                    drive.set_kinematics(linear_speed, rotation_speed)

                # Parameter updates (wheel geometry, free-run period)
                elif action == 'parameter':
                    wheel_diameter_mm = cmd.get('wheel_diameter_mm', None)
                    wheel_base_mm     = cmd.get('wheel_base_mm', None)
                    new_free_ping_period = cmd.get('free_ping_period', None)

                    if wheel_diameter_mm is not None:
                        drive.set_wheel_diameter(wheel_diameter_mm)
                        display.write(0, 'Set wheel diam')

                    if wheel_base_mm is not None:
                        drive.set_wheel_base(wheel_base_mm)
                        display.write(0, 'Set wheel base')

                    if new_free_ping_period is not None:
                        new_free_ping_period = max(minimum_free_range_period, new_free_ping_period)
                        set_free_run(new_free_ping_period, state)

                # Discrete move/turn steps
                elif action == 'step':
                    display.write(0, 'step')
                    rotation_speed = cmd.get('rotation_speed', 0)
                    linear_speed = cmd.get('linear_speed', 0)
                    distance = cmd.get('distance', 0)
                    angle = cmd.get('angle', 0)

                    if abs(angle) > 0 and rotation_speed == 0: rotation_speed = 90
                    if abs(distance) > 0 and linear_speed == 0: linear_speed = 0.1

                    if abs(angle) > 0: drive.turn_angle(angle, rotation_speed)
                    if abs(angle) > 0 and abs(distance) > 0: time.sleep(0.1)
                    if abs(distance) > 0: drive.drive_distance(distance, linear_speed)

                # Acoustic actions: 'ping' (emit→gate→capture) or 'listen' (capture only)
                elif action in ['ping', 'listen']:
                    # Extract parameters for acquisition
                    sample_rate = cmd.get('sample_rate')
                    samples = cmd.get('samples')
                    # ── Pause free-run to prevent any overlap with acquisition ──
                    prev_period = state['period']
                    if prev_period > 0:
                        # Small guard from the *last* free-run pulse to let ringing settle
                        since_last = ticks_diff(ticks_ms(), state['last_mark'])
                        if since_last < measure_guard_ms: time.sleep_ms(int(measure_guard_ms - since_last))
                        # Disable free-run during the acquisition
                        set_free_run(0, state)

                    # ── Do the acquisition ──
                    display.write(0, action)
                    buf0, buf1, buf2, timing_info = snr.acquire(action, sample_rate, samples)
                    packed = bytes(buf0) + bytes(buf1) + bytes(buf2)

                    response = {}
                    response['data'] = packed
                    response['timing_info'] = timing_info
                    response['mode'] = action
                    print(f'[{action}] {timing_info}')
                    bridge.send_data(response)

                    if prev_period > 0: set_free_run(prev_period, state)

                elif action == 'acknowledge':
                    if verbose: print('[Main] Acknowledgment received')

        # ── Free-running pulsing (drift-free absolute schedule) ──
        if state['next_due'] is not None and ticks_diff(now, state['next_due']) >= 0:
            # Emit a free-run pulse (no capture)
            led.toggle_color(0)
            snr.acquire('pulse')
            # Measure achieved interval (edge-to-edge) for telemetry
            this_mark = ticks_ms()
            interval  = ticks_diff(this_mark, state['last_mark'])
            pulse_avg.add(interval)
            state['last_mark'] = this_mark

            # March schedule forward exactly one period; skip missed slots if late
            state['next_due'] = ticks_add(state['next_due'], state['period'])
            while ticks_diff(now, state['next_due']) >= state['period']:
                state['next_due'] = ticks_add(state['next_due'], state['period'])

            # Print/display sparingly to avoid jitter
            if pulse_avg.total_samples() % 200 == 0:
                avg = pulse_avg.average()
                display.write(4, f'P avg: {avg:.1f}ms')
                pulse_avg.reset_total_count()

        # ── LED heartbeat (non-blocking) ──
        if ticks_diff(now, last_heartbeat) >= heartbeat_interval:
            led.toggle_color(2)
            last_heartbeat = now
