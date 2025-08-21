import screen
import leds
import sonar
import time
import motors
import wifi
import settings
import bumpers

selected_tests = [5]

if 0 in selected_tests:
    # ───────────────────── Wifi Test ─────────────────────
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
        print('IP address:', ip)


if 1 in selected_tests:
    # ───────────────────── LED Test ─────────────────────
    print('[LEDs] Running LED test...')
    leds = leds.LEDs()
    for i in range(6):
        leds.set(i, 'blue')
        time.sleep(0.5)
    for i in range(6):
        leds.set(i, 'off')
        time.sleep(0.5)
    print('[LEDs] LED test completed.')

if 2 in selected_tests:
    # ───────────────────── Screen Test ─────────────────────
    print('[Screen] Running screen test...')
    display = screen.Screen()
    for i in range(5):
        display.clear()
        time.sleep(1)
        display.write(0, 'Screen Test')
        display.write(1, 'Hello, World!')
        time.sleep(1)
    print('[Screen] Screen test completed.')

if 3 in selected_tests:
    # ───────────────────── Bumper Test ─────────────────────
    start_time = time.ticks_ms()
    bump = bumpers.Bumpers()
    while True:
        bump_left, bump_right = bump.read()
        print(f'Left: {bump_left}, Right: {bump_right}   ', end='\r')
        current_time = time.ticks_ms()
        elapsed_time = time.ticks_diff(current_time, start_time)
        if elapsed_time > 10000: break

if 4 in selected_tests:
    # ───────────────────── Sonar Test ─────────────────────
    print('[Sonar] Running sonar test...')
    for i in range(5):
        buf0, buf1, buf2, timing_info = sonar.measure(10000, 200)
        max_buf0 = max(buf0) # Maximum value of emitter
        max_buf1 = max(buf1) # Maximum value of receiver 1
        max_buf2 = max(buf2) # Maximum value of receiver 2
        print(f'Pulse {i} => Max values: EMIT={max_buf0}, RCV1={max_buf1}, RCV2={max_buf2}')
        time.sleep(0.5)
    print('[Sonar] Sonar test completed.')

if 5 in selected_tests:
    # ───────────────────── Motors Test ─────────────────────
    print('[Motors] Running motors test...')
    drive = motors.Motors()
    for i in range(5):
        drive.turn_angle(-10)
        time.sleep(1)
        drive.turn_angle(10)
        time.sleep(1)
    print('[Motors] Motors test completed.')
