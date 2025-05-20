import screen
import leds
import maxbotix
import time
import motors
import wifi
import settings

print('[Main] Initialize')

display = screen.Screen()
led = leds.LEDs()
sonar = maxbotix.MaxBotix()
drive = motors.Motors()
bridge = wifi.WifiServer()

display.write(0, 'Booting')
led.set_all('off')
led.set(0, 'orange')
led.set(1, 'orange')
led.set(2, 'orange')

print("[Connecting] Enabling ESP8266...")
bridge.enable()
print("[Connecting] Connecting to Wi-Fi...")
bridge.connect_wifi("batnet", "lebowski")
led.set(0, 'green')

print("[Connecting] Starting server...")
bridge.start_server(1234)
led.set(1, 'green')

print("[Connecting] Getting IP address...")
ip = bridge.get_ip()
display.write(1, ip)
led.set(2, 'green')

print('[Main] Starting main loop')

loop_nr = 0
while True:
    command = bridge.wait_command()
    print('[Received]', command, loop_nr)
    
    if loop_nr == 0: led.set(5, 'blue')
    if loop_nr == 1: led.set(5, 'orange')
    
    parsed = wifi.parse_command(command)
    action = parsed[0]
    values = parsed[1]
    
    print('[Parsed]', parsed)
    
    if action == 'motors':
        display.write(2, action)
        drive.set_speeds(values[0], values[1])
    
    if action == 'ping':
        display.write(2, action)
        b1, b2 = sonar.measure(values[0], values[1])
        byte_data1 = wifi.array_to_bytes(b1)
        byte_data2 = wifi.array_to_bytes(b2)
        byte_data_all = byte_data1 + byte_data2
        bridge.send_data(data=byte_data_all)
        
    loop_nr = loop_nr + 1
    if loop_nr == 2: loop_nr = 0
   



