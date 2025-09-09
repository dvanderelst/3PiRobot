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

# tick helpers (wrap once, use everywhere)
ticks_ms   = time.ticks_ms
ticks_add  = time.ticks_add
ticks_diff = time.ticks_diff

def set_free_run(val_ms, state):
    """Enable/disable free-run pulser by setting period and next deadline."""
    val = max(0, int(val_ms))
    state['period'] = val
    if val > 0:
        now = ticks_ms()
        state['last_mark'] = now
        state['next_due'] = ticks_add(now, val)
        state['display'].write(4, f'Free: {val}ms')
    else:
        state['next_due'] = None
        state['display'].write(4, 'Free: off')

def regrid_free_run(state):
    """Re-align schedule after a blocking action (e.g., manual ping)."""
    if state['period'] > 0:
        now = ticks_ms()
        state['last_mark'] = now
        state['next_due']  = ticks_add(now, state['period'])

def main(selected_ssid=None):
    beeper = beeps.Beeper()

    # ───────────────────── Init ─────────────────────
    beeper.play('startup_proc')
    print('[Main] Initializing systems...')

    verbose = settings.verbose
    led = leds.LEDs()
    display = screen.Screen()
    bump = bumpers.Bumpers(leds=led)
    drive = motors.Motors()
    pulse_avg = utils.RollingAverage(10)

    display.clear()
    display.write(0, 'Systems up')
    led.set_all('off')

    # ───────────────────── Wi-Fi ─────────────────────
    bridge, ip, ssid = wifi.setup_wifi(ssids=selected_ssid)
    beeper.play('wifi_connected'); led.set(0, 'green')
    display.clear()
    display.write(0, 'Connected')
    display.write(1, ssid or '')
    display.write(2, ip or '')

    # ───────────────────── Startup pulses ─────────────────────
    print("[Main] Performing startup pulses...")
    for _ in range(5):
        led.set(2, 'red'); sonar.pulse()
        time.sleep(0.05)
        led.set(2, 'off')
        time.sleep(0.05)

    # ───────────────────── Main loop ─────────────────────
    print('[Main] Entering main loop...')
    current_led_color = 'blue'
    led.set(2, current_led_color)

    last_toggle = ticks_ms()
    toggle_interval = 1000  # ms
    command_queue = []

    display.write(0, 'Ready')
    beeper.play('main_loop')

    # ── Free-run pulser state ──
    state = {
        'period': 0,          # ms; 0 = disabled
        'next_due': None,     # absolute deadline
        'last_mark': ticks_ms(),
        'display': display,
    }
    set_free_run(0, state)  # start disabled

    while True:
        # ── Wi-Fi commands ──
        new_cmds = bridge.read_messages()
        if new_cmds:
            command_queue.extend(new_cmds)

        # ── Bumpers ──
        bump_left, bump_right = bump.read()
        if bump_left or bump_right:
            drive.stop()

        # ── Commands ──
        if command_queue:
            cmds = command_queue
            command_queue = []
            if verbose:
                print(f"[Main] Received: {cmds}")
            for cmd in cmds:
                action = cmd.get('action')

                if action == 'kinematics':
                    display.write(0, 'kinematics')
                    rotation_speed = cmd.get('rotation_speed')
                    linear_speed = cmd.get('linear_speed')
                    drive.set_kinematics(linear_speed, rotation_speed)

                elif action == 'parameter':
                    wheel_diameter_mm = cmd.get('wheel_diameter_mm', None)
                    wheel_base_mm = cmd.get('wheel_base_mm', None)
                    new_free_ping_interval = cmd.get('free_ping_interval', None)
                    if wheel_diameter_mm is not None:
                        drive.set_wheel_diameter(wheel_diameter_mm)
                        display.write(0, 'Set wheel diam')
                    if wheel_base_mm is not None:
                        drive.set_wheel_base(wheel_base_mm)
                        display.write(0, 'Set wheel base')
                    if new_free_ping_interval is not None:
                        set_free_run(new_free_ping_interval, state)

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

                elif action in ['ping', 'listen']:
                    wait_for_emission = True
                    if action == 'listen': wait_for_emission = False
                    display.write(0, action)
                    sample_rate = cmd.get('sample_rate')
                    samples = cmd.get('samples')

                    led.set(0, 'red')
                    buf0, buf1, buf2, timing_info = sonar.measure(sample_rate, samples, wait_for_emission)
                    led.set(0, 'green')

                    # keep free-run on-grid after blocking measurement
                    regrid_free_run(state)

                    response = {
                        'action': 'ping_response',
                        'data': [list(buf0), list(buf1), list(buf2)],
                        'timing_info': timing_info
                    }
                    bridge.send_data(response)

                elif action == 'acknowledge':
                    if verbose:
                        print('[Main] Acknowledgment received')

        now = ticks_ms()

        # ── Free-running pulsing (absolute schedule) ──
        if state['next_due'] is not None and ticks_diff(now, state['next_due']) >= 0:
            sonar.pulse()

            # measure achieved interval (edge-to-edge)
            this_mark = ticks_ms()
            interval = ticks_diff(this_mark, state['last_mark'])
            pulse_avg.add(interval)
            state['last_mark'] = this_mark

            # march schedule forward by exactly one period; skip missed slots if late
            state['next_due'] = ticks_add(state['next_due'], state['period'])
            while ticks_diff(now, state['next_due']) >= state['period']:
                state['next_due'] = ticks_add(state['next_due'], state['period'])

            # sparse telemetry
            if pulse_avg.total_samples() % 100 == 0:
                average = pulse_avg.average()
                print(f'Target: {state["period"]}, Avg: {average:.1f}')
                display.write(4, f'P avg: {average:.1f}ms')
                pulse_avg.reset_total_count()

        # ── LED heartbeat ──
        if ticks_diff(now, last_toggle) >= toggle_interval:
            current_led_color = 'blue' if current_led_color == 'orange' else 'orange'
            led.set(2, current_led_color)
            last_toggle = now
