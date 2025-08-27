import screen
# import leds
# import sonar
# import time
# import motors
# import wifi
# import settings
# import bumpers


def wifi_test():
    import wifi
    import screen
    print('[WiFi] Running WiFi test...')
    bridge, ip, ssid = wifi.setup_wifi()
    print('[WiFi] WiFi test complete.')
    display = screen.Screen()
    display.clear()
    display.write(0, 'Connected')
    display.write(1, ssid)
    display.write(2, ip)

def motors_test():
    import motors
    import time
    import screen
    print('[Motors] Running motors test...')
    print('Make sure the robot is switched on! Blue led on left side should be lit.')
    display = screen.Screen()
    display.clear()
    display.write(0, 'Motors test')
    display.write(1, 'will start...')
    time.sleep(3)
    drive = motors.Motors()
    for i in range(5):
        drive.turn_angle(-10)
        time.sleep(1)
        drive.turn_angle(10)
        time.sleep(1)
    print('[Motors] Motors test completed.')

def bumpers_test():
    import bumpers
    import time
    import screen
    print('[Bumpers] Running bumpers test...')
    print('Press the bumpers to see the response on the screen.')
    display = screen.Screen()
    display.clear()
    display.write(0, 'Bumpers test')
    bump = bumpers.Bumpers()
    while True:
        states = bump.read()
        display.write(1, f'L: {states[0]} R: {states[1]}')
        # print the states to the console. Reuse/overwrite the same line.
        print(f'\rLeft bumper: {states[0]} | Right bumper: {states[1]}      ', end='')
        time.sleep(0.1)


def sonar_test():
    import sonar
    import time
    import screen
    print('[Sonar] Running sonar test...')
    print('Make sure there is an object in front of the sonar sensor.')
    display = screen.Screen()
    display.clear()
    display.write(0, 'Sonar test')
    while True:
        buf0, buf1, buf2, timing_info = sonar.measure(10000, 200)
        max_emitter = max(buf0)
        max_sensor1 = max(buf1)
        max_sensor2 = max(buf2)
        print(f'\rE:{max_emitter}, S1:{max_sensor1}, S2:{max_sensor2}          ', end='')
        display.write(0, f'E:{max_emitter}')
        display.write(1, f'S1:{max_sensor1}')
        display.write(2, f'S2:{max_sensor2}')
        time.sleep(1)

