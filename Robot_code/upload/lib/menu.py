# Minimal menu for Pololu 3pi+ 2040: A=prev, B=select, C=next
from pololu_3pi_2040_robot.display import Display
from pololu_3pi_2040_robot.buzzer import Buzzer
from pololu_3pi_2040_robot.buttons import ButtonA, ButtonB, ButtonC
import time, sys

display = Display()
buzzer  = Buzzer()
btnA, btnB, btnC = ButtonA(), ButtonB(), ButtonC()

def clear_display():
    display.fill(0)
    display.show()

def msg_wait(message, expect=None, timeout=None):
    """
    Show a message and wait for a button press.
    - expect: 'A', 'B', 'C' or None (accept any)
    - timeout: seconds to wait, or None = wait forever
    Returns: 'A'/'B'/'C' if pressed, or None on timeout
    """
    display.fill(0); display.text(message, 0, 0); display.show()
    last = {"A": False, "B": False, "C": False}
    start = time.time()
    while True:
        if timeout is not None and (time.time() - start) > timeout: clear_display(); return None
        for btn, key, beep in [(btnA,"A","!c32"), (btnB,"B","!e32"), (btnC,"C","!g32")]:
            cur = btn.is_pressed()
            if cur and not last[key]:  # rising edge
                buzzer.play(beep)
                if expect is None or key == expect:
                    clear_display()
                    # wait until ALL buttons are released before returning
                    while btnA.is_pressed() or btnB.is_pressed() or btnC.is_pressed(): time.sleep(0.02)
                    time.sleep(0.05)  # tiny debounce cushion
                    return key
            last[key] = cur
        time.sleep(0.02)

class Menu:
    def __init__(self, items, title="Select", footer="A:UP  B:Select  C:DOWN"):
        self.items = list(items)
        self.title = title
        self.footer = footer
        self.i = 0
        self._last = {"A": False, "B": False, "C": False}

    def _pressed(self, btn, key):
        cur = btn.is_pressed()
        was = self._last[key]
        self._last[key] = cur
        return (not was) and cur  # rising edge

    def _draw(self):
        display.fill(0)
        y = 0
        if self.title:
            display.text(self.title, 0, y)
            y += 12
        for k, name in enumerate(self.items):
            prefix = ">" if k == self.i else " "
            display.text(prefix + name, 0, y + 10*k)
        if self.footer:
            display.text(self.footer, 0, 50)  # small footer near bottom
        display.show()

    def show(self, timeout=None):
        """Blocks until selection or timeout.
        Returns selected item (string) or None on timeout.
        """
        start = time.time()
        self._draw()
        while True:
            if timeout is not None and (time.time() - start) > timeout:
                return None

            if self._pressed(btnA, "A"):           # previous
                self.i = (self.i - 1) % len(self.items)
                buzzer.play("!c32")
                self._draw()

            if self._pressed(btnC, "C"):           # next
                self.i = (self.i + 1) % len(self.items)
                buzzer.play("!g32")
                self._draw()

            if self._pressed(btnB, "B"):           # select
                buzzer.play("!e32")
                time.sleep(0.12)                   # tiny debounce
                return self.items[self.i]

            time.sleep(0.02)

def run_choice(choice):
    """Helper to run a chosen .py, 'exit to REPL', or ignore None."""
    if not choice:
        return
    if choice == "exit to REPL":
        sys.exit(0)
    if choice.endswith(".py"):
        __import__(choice[:-3])
