import maxbotix
import time

sonar = maxbotix.MaxBotix()

while True:
    print('measure')
    b1, b2 = sonar.measure(10000, 200)
    time.sleep(1)
    
