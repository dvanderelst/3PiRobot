# boot.py runs before main.py at every MicroPython startup.
# Drive the three MB1360 sonar trigger lines low immediately, before any
# slower app code (Wi-Fi association, etc.) has a chance to run. This
# suppresses sensor free-run mode during the boot window, which is a
# candidate cause of the bistate calibration we're investigating.
#
# GP22 — emitter trigger
# GP24 — recv2 trigger
# GP7  — recv1 trigger (shared with buzzer; my_pololu_lib/buzzer.py
#        is patched to release this pin between beeps so the boot-time
#        pull-down stays effective).

from machine import Pin
Pin(22, Pin.OUT, value=0)
Pin(24, Pin.OUT, value=0)
Pin(7,  Pin.OUT, value=0)
