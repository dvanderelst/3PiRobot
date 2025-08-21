import screen
import leds
import sonar
import time
import motors
import wifi
import settings
import bumpers


def select_tests():
    print('==================== Test Selection ==================')
    print("Select tests to run:")
    print("0: WiFi Test")
    print("1: LED Test")
    print("2: Screen Test")
    print("3: Bumper Test")
    print("4: Motor Test")
    print("5: Sonar Test")
    print("Enter test numbers separated by commas (e.g., 0,2,4): ")

    user_input = input()
    selected_tests = [int(x) for x in user_input.split(',') if x.isdigit() and 0 <= int(x) <= 5]

    return selected_tests

def run_tests(selected_tests):
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
        lights = leds.LEDs()
        for i in range(6):
            lights.set(i, 'blue')
            time.sleep(0.5)
        for i in range(6):
            lights.set(i, 'off')
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
        # ───────────────────── Motors Test ─────────────────────
        print('[Motors] Running motors test...')
        print('Make sure the robot is switched on! Blue led on left side should be lit.')
        drive = motors.Motors()
        for i in range(5):
            drive.turn_angle(-10)
            time.sleep(1)
            drive.turn_angle(10)
            time.sleep(1)
        print('[Motors] Motors test completed.')

    if 5 in selected_tests:
        # ───────────────────── Sonar Test ─────────────────────
        print('[Sonar] Running sonar test...')
        for i in range(100):
            buf0, buf1, buf2, timing_info = sonar.measure(10000, 200)
            max_buf0 = max(buf0) # Maximum value of emitter
            max_buf1 = max(buf1) # Maximum value of receiver 1
            max_buf2 = max(buf2) # Maximum value of receiver 2
            print(f'Pulse {i} => Max values: EMIT={max_buf0}, RCV1={max_buf1}, RCV2={max_buf2}')
            time.sleep(0.6)
        print('[Sonar] Sonar test completed.')

