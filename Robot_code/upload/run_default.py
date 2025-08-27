import screen
import leds
import sonar
import time
import motors
import wifi
import settings
import bumpers
import beeps

beeper = beeps.Beeper()

# ───────────────────── Initialization ─────────────────────
beeper.play('startup_proc')
print('[Main] Initializing systems...')

verbose = settings.verbose

led = leds.LEDs()
display = screen.Screen()
bump = bumpers.Bumpers(leds=led)
drive = motors.Motors()

display.write(0, 'Systems up')
led.set_all('off')

# ───────────────────── Wi-Fi Setup ─────────────────────
bridge, ip, ssid = wifi.setup_wifi()
if bridge is None: beeper.play('error')
if bridge is not None: beeper.play('wifi_connected'); led.set(0, 'green')
display.clear()
display.write(0, 'Connected')
display.write(1, ssid)
display.write(2, ip)

# ───────────────────── Initial LED Blink ─────────────────────
print("[Main] Performing startup pulses...")
for i in range(5):
    led.set(2, 'red')
    sonar.pulse()
    time.sleep(0.05)
    led.set(2, 'off')
    time.sleep(0.05)

# ───────────────────── Main Loop ─────────────────────
print('[Main] Entering main loop...')

loop_nr = 0
current_led_color = 'blue'
led.set(2, current_led_color)

last_toggle = time.ticks_ms()
toggle_interval = 500  # milliseconds

command_queue = []

display.write(0, 'Ready')
beeper.play('main_loop')

while True:
    # # ── Wi-Fi Commands ──
    new_commands = bridge.read_messages()
    if new_commands: command_queue.extend(new_commands)
    # # ── Bumper Check ──
    bump_left, bump_right = bump.read()
    if bump_left or bump_right: drive.stop()
    # ── Command Processing ──
    if command_queue:
        commands = command_queue.copy()
        command_queue.clear()
        if verbose:
            print(f"[Main] Received: {commands}")
        for cmd in commands:
            action = cmd.get('action')

            if action == 'kinematics':
                display.write(0, 'kinematics')
                rotation_speed = cmd.get('rotation_speed')
                linear_speed = cmd.get('linear_speed')
                drive.set_kinematics(linear_speed, rotation_speed)

            elif action == 'parameter':
                wheel_diameter_mm = cmd.get('wheel_diameter_mm', None)
                wheel_base_mm = cmd.get('wheel_base_mm', None)
                if wheel_diameter_mm is not None:
                    drive.set_wheel_diameter(wheel_diameter_mm)
                    display.write(0, 'Set wheel diameter')
                if wheel_base_mm is not None:
                    drive.set_wheel_base(wheel_base_mm)
                    display.write(0, 'Set wheel base')

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

            elif action == 'ping':
                display.write(0, 'ping')
                sample_rate = cmd.get('sample_rate')
                samples = cmd.get('samples')
                buf0, buf1, buf2, timing_info = sonar.measure(sample_rate, samples)
                response = {
                    'action': 'ping_response',
                    'data': [list(buf0), list(buf1), list(buf2)],
                    'timing_info': timing_info
                }
                bridge.send_data(response)

            elif action == 'acknowledge':
                if verbose: print('[Main] Acknowledgment received')

    # ── LED Blinking Indicator ──
    now = time.ticks_ms()
    if time.ticks_diff(now, last_toggle) >= toggle_interval:
        current_led_color = 'blue' if current_led_color == 'orange' else 'orange'
        led.set(2, current_led_color)
        last_toggle = now

    time.sleep(0.01)
