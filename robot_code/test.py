import my_leds
import my_maxbotix
import my_motors
import time

sonar = my_maxbotix.MaxBotix()
motors = my_motors.Motors()
start = time.tic