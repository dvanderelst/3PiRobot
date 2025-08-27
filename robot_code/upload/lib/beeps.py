from pololu_3pi_2040_robot.buzzer import Buzzer as _Buzzer

class Beeper:
    """Wrapper around Pololu Buzzer with predefined UI beeps and custom play."""

    SOUNDS = {
        # General feedback
        "ok":      "!c16",                     # quick low beep
        "error":   "!c16 !c16",                # double beep
        "select":  "!e32",                     # short high blip
        "back":    "!g32",                     # short low blip
        "done":    "O6 g16 e16 c16",           # descending (finished)
        "alert":   "O5 c8 c8 c8",              # three short beeps
        "success": "O5 c16 e16 g8",            # arpeggio up
        "fail":    "O4 g16 e16 c8",            # arpeggio down
        "powerup": "O5 c32 d32 e32 g16",       # sweep up
        "powerdn": "O5 g32 e32 d32 c16",       # sweep down

        # Event jingles
        "robot_start":  "O5 c16 e16 g16",                # major chord (already had)
        "startup_proc": "O5 c8 d8 e8",                   # ascending three notes
        "wifi_connected": "O6 c16 g16 c8",
        "main_loop":    "O5 e32 g32 c16 e16",            # repeating motif, “loop” feel
    }

    def __init__(self):
        self._bz = _Buzzer()

    def play(self, name_or_song):
        """
        Play a predefined sound by key (e.g. 'ok', 'error')
        or a custom song string.
        """
        song = self.SOUNDS.get(name_or_song, name_or_song)
        self._bz.play(song)

    def off(self):
        """Stop any sound currently playing."""
        self._bz.off()
