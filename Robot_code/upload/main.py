import menu
import testing
import beeps
import default_algorithm

b = beeps.Beeper()
b.play("robot_start")
menu_items = ['0: Use Batnet', '1: WiFi Test', '2: Motors Test', '3: Bumpers Test', '4: Sonar Test']
key_pressed = menu.msg_wait('Press B for test menu', expect='B', timeout=3)
print('button press:', key_pressed)

if key_pressed == 'B':
    m = menu.Menu(items=menu_items, footer='')
    selected = m.show()
    selected_nr = int(selected[0])
    print(selected, selected_nr)
    if selected_nr == 0: default_algorithm.main(['batnet'])
    if selected_nr == 1: testing.wifi_test()
    if selected_nr == 2: testing.motors_test()
    if selected_nr == 3: testing.bumpers_test()
    if selected_nr == 4: testing.sonar_test()
else:
    print('No button pressed, skipping test menu.')
    ssids = ['batnet_robotlab', 'ZyXEL39940']
    default_algorithm.main(ssids)
