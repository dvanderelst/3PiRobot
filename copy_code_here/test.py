import my_leds
import my_maxbotix
import time

sonar = my_maxbotix.MaxBotix()

for x in range(10):
    sonar.measure()
    time.sleep(0.1)
