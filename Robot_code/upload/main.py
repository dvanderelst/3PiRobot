import menu
import testing
import beeps

b = beeps.Beeper()
b.play("robot_start")
menu_items = ['0: WiFi Test', '1: Motors Test', '2: Bumpers Test', '3: Sonar Test']
key_pressed = menu.msg_wait('Press B for test menu', expect='B', timeout=3)
print('button press:', key_pressed)
if key_pressed == 'B':
    m = menu.Menu(items=menu_items)
    selected = m.show()
    selected_nr = int(selected[0])
    print(selected, selected_nr)
    if selected_nr == 0: testing.wifi_test()
    if selected_nr == 1: testing.motors_test()
    if selected_nr == 2: testing.bumpers_test()
    if selected_nr == 3: testing.sonar_test()
else:
    print('No button pressed, skipping test menu.')
    import default_algorithm
