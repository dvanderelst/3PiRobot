import screen
import leds
import maxbotix
import time
import motors
import wifi
import settings
import bumpers

verbose = settings.verbose

print('[Main] Initialize')
# ────────────── Init systems ──────────────

led = leds.LEDs()
sonar = maxbotix.MaxBotix()
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
command_queue = []
display.write(0, 'Ready')
while True:
    # 1. Check for new Wi-Fi commands
    new_commands = bridge.check_commands()  # Non-blocking
    if new_commands: command_queue.extend(new_commands)

    # 2. Read bumpers
    left, right = bump.read()
    if left or right:
        print("[BUMP] L:", left, "R:", right)
        display.write(3, f"BUMP L:{int(left)} R:{int(right)}")
        drive.set_speeds(0, 0)

    # 3. Blink LED to indicate loop activity
    #led.set(5, 'blue' if loop_nr == 0 else 'orange')
    loop_nr = (loop_nr + 1) % 2

    # 4. Process incoming commands
    if command_queue:
        commands = command_queue.copy()
        command_queue.clear()
        if verbose: print('[Received]', commands)

        parsed = wifi.parse_commands(commands)
        for action, values in parsed:
            if verbose > 1:
                print('[Parsed]', (action, values))

            if action == 'motors':
                display.write(0, action)
                drive.set_speeds(values[0], values[1])

            elif action == 'ping':
                display.write(0, action)
                b1, b2 = sonar.measure(values[0], values[1])
                data = wifi.array_to_bytes(b1) + wifi.array_to_bytes(b2)
                bridge.send_data(data)

    time.sleep(0.01)
