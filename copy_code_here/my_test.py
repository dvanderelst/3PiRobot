from pololu_3pi_2040_robot import robot
import Speed

rgb_leds = robot.RGBLEDs()
display = robot.Display()
motors = robot.Motors()

converter = Speed.Converter()


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

cps = converter.mps2cps(0.1)

motors.set_speeds(cps, cps)
