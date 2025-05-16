import screen
import leds
import maxbotix
import time
import motors


display = screen.Screen()
led = leds.LEDs()
sonar = maxbotix.MaxBotix()
drive = motors.Motors()

drive.set_speeds(0.1)
for x in range(100):
    display.write(0, x)
    sonar.measure()
    time.sleep(0.025)
    
drive.stop()
