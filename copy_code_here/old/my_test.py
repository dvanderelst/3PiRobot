from pololu_3pi_2040_robot import robot
import Speed
import time

rgb_leds = robot.RGBLEDs()
display = robot.Display()

controller = Speed.Controller()


# for led in range(6):
#     rgb_leds.set(led, [0, 0, 0])
# rgb_leds.show()
# 
# for led in range(7):
#     rgb_leds.set(led, [100, 0, 0])
#     rgb_leds.show()

# display.fill(0)
# display.text("A", 0, 0)
# display.text("B", 0, 10)
# display.text("C", 0, 23)
# display.text("D", 0, 33)
# display.text("E", 0, 47)
# display.text("F", 0, 57)
# display.show()



controller.set_mps(0.1, 0.3)
time.sleep(1)
controller.set_mps(0.1,