import screen
import leds
import sonar
import time
import motors
import wifi
import settings
import bumpers

verbose = settings.verbose

print('[Main] Initialize')
# ────────────── Init systems ──────────────

led = leds.LEDs()
display = screen.Screen()
bump = bumpers.Bumpers()
drive = motors.Motors()
display.write(0, 'Systems up')
led.set_all('off')

# ────────────── Wi-Fi Setup ──────────────
print("[Main] Preparing Wi-Fi module...")
bridge = wifi.WifiServer()
bridge.setup()

print("[Main] Connecting to Wi-Fi...")
ssid, password = settings.ssid_list[settings.ssid_index]
join_response = bridge.connect_wifi(ssid, password)

if "ERROR" in join_response:
    print(f"[FAIL] Could not join network '{ssid}'")
else:
    print(f"[OK] Successfully joined '{ssid}'")
    print("→ IP info:")
    print(join_response)
    ip = bridge.get_ip()
    display.write(0, 'Connected')
    display.write(1, ssid)
    display.write(2, ip)
    led.set(0, 'green')

# ────────────── Server Startup ──────────────
print("[Main] Starting server...")
bridge.start_server(1234)
display.write(0, 'Server up')

# ──────────── Starting main loop ────────────
print('[Main] Starting main loop')
loop_nr = 0
current_led_color = 'blue'
led.set(5, current_led_color)

last_toggle = time.ticks_ms()
toggle_interval = 500  # milliseconds

command_queue = []
display.write(0, 'Ready')

while True:
    # 1. Check for new Wi-Fi commands
    new_commands = bridge.read_messages()
    if new_commands: command_queue.extend(new_commands)

    # 2. Read bumpers
    left, right = bump.read()
    if left or right:
        print("[BUMP] L:", left, "R:", right)
        display.write(3, f"BUMP L:{int(left)} R:{int(right)}")
        drive.set_speeds(0, 0)

    # 3. Process incoming commands
    if command_queue:
        commands = command_queue.copy()
        command_queue.clear()
        if verbose: print('[Main] Received:', commands)

        for cmd in commands:
            action = cmd.get('action')

            if action == 'motors':
                display.write(0, action)
                drive.set_speeds(cmd.get('left', 0), cmd.get('right', 0))

            elif action == 'ping':
                display.write(0, action)
                rate = cmd.get('rate')
                samples = cmd.get('samples')
                buf0, buf1, buf2, timing_info = sonar.measure(rate, samples)
                package = {'action': 'ping_response', 'data': [list(buf0), list(buf1), list(buf2)], 'timing_info': timing_info}
                bridge.send_data(package)

            elif action == 'acknowledge':
                if verbose: print('[Main] Acknowledgment received')

    # 4. Blink LED 5 to indicate readiness
    now = time.ticks_ms()
    if time.ticks_diff(now, last_toggle) >= toggle_interval:
        current_led_color = 'blue' if current_led_color == 'orange' else 'orange'
        led.set(5, current_led_color)
        last_toggle = now

    time.sleep(0.01)
