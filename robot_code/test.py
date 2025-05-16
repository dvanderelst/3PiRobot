import command_listener
import screen
import leds

listener = command_listener.CommandListener()
screen = screen.Screen()
leds = leds.LEDs()

screen.write(0, 'awaiting command')
leds.set_all('blue')

x = listener.wait_for_command()
print(x)