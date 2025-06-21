import screen
import leds
import sonar
import time
import motors
import wifi
import settings
import bumpers

# ───────────────────── Initialization ─────────────────────
print('[Main] Initializing systems...')

verbose = settings.verbose

led = leds.LEDs()
display = screen.Screen()
bump = bumpers.Bumpers()
drive = motors.Motors()

display.write(0, 'Systems up')
led.set_all('off')

# ───────────────────── Wi-Fi Setup ─────────────────────
print("[WiFi] Preparing module...")
bridge = wifi.WifiServer()
bridge.setup()

print("[WiFi] Connecting...")
ssid, password = settings.ssid_list[settings.ssid_index]
join_response = bridge.connect_wifi(ssid, password)

if "ERROR" in join_response:
    print(f"[WiFi] [FAIL] Could not join network '{ssid}'")
else:
    print(f"[WiFi] [OK] Connected to '{ssid}'")
    print("        → IP info:")
    print(join_response)

    ip = bridge.get_ip()
    display.write(0, 'Connected')
    display.write(1, ssid)
    display.write(2, ip)
    led.set(0, 'green')

# ───────────────────── Server Setup ─────────────────────
print("[Main] Starting command server...")
bridge.start_server(1234)
display.write(0, 'Server up')

# ───────────────────── Initial LED Blink ─────────────────────
print("[Main] Performing startup pulses...")
for i in range(5):
    led.set(5, 'red')
    sonar.pulse()
    time.sleep(0.05)
    led.set(5, 'off')
    time.sleep(0.05)

# ───────────────────── Main Loop ─────────────────────
print('[Main] Entering main loop...')
loop_nr = 0
current_led_color = 'blue'
led.set(5, current_led_color)

last_toggle = time.ticks_ms()
toggle_interval = 500  # milliseconds

command_queue = []
display.write(0, 'Ready')

while True:
    # ── Wi-Fi Commands ──
    new_commands = bridge.read_messages()
    if new_commands:
        command_queue.extend(new_commands)

    # ── Bumper Check ──
    bump_left, bump_right = bump.read()
    if bump_left or bump_right:
        print(f"[BUMP] Left: {bump_left} | Right: {bump_right}")
        display.write(3, f"BUMP L:{int(bump_left)} R:{int(bump_right)}")
        drive.set_speeds(0, 0)

    # ── Command Processing ──
    if command_queue:
        commands = command_queue.copy()
        command_queue.clear()
        if verbose:
            print(f"[Main] Received: {commands}")

        for cmd in commands:
            action = cmd.get('action')

            if action == 'motors':
                display.write(0, 'motors')
                motor_left = cmd.get('left', 0)
                motor_right = cmd.get('right', 0)
                drive.set_speeds(motor_left, motor_right)

            elif action == 'kinematics':
                display.write(0, 'kinematics')
                rotation_speed = cmd.get('rotation_speed')
                linear_speed = cmd.get('linear_speed')
                drive.set_kinematics(linear_speed, rotation_speed)

            elif action == 'step':
                display.write(0, 'step')
                rotation_speed = cmd.get('rotation_speed', 0)
                linear_speed = cmd.get('linear_speed', 0)
                distance = cmd.get('distance', 0)
                angle = cmd.get('angle', 0)
                
                if abs(angle) > 0 and  == 0: rotation_speed = 90
                if abs(distance) > 0 and linear_speed == 0: linear_speed = 0.1
                            
                drive.turn_angle(angle, rotation_speed)
                time.sleep(0.1)
                drive.drive_distance(distance, linear_speed)

            elif action == 'ping':
                display.write(0, 'ping')
                rate = cmd.get('rate')
                samples = cmd.get('samples')
                buf0, buf1, buf2, timing_info = sonar.measure(rate, samples)
                response = {
                    'action': 'ping_response',
                    'data': [list(buf0), list(buf1), list(buf2)],
                    'timing_info': timing_info
                }
                bridge.send_data(response)

            elif action == 'acknowledge':
                if verbose:
                    print('[Main] Acknowledgment received')

    # ── LED Blinking Indicator ──
    now = time.ticks_ms()
    if time.ticks_diff(now, last_toggle) >= toggle_interval:
        current_led_color = 'blue' if current_led_color == 'orange' else 'orange'
        led.set(5, current_led_color)
        last_toggle = now

    time.sleep(0.01)
