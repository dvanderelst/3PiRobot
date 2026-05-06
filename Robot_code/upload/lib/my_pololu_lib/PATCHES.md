# Local patches to vendored Pololu 3pi+ 2040 library

This folder is a vendored copy of Pololu's MicroPython library for the
3pi+ 2040 robot (originally `pololu_3pi_2040_robot`). It has been renamed
to `my_pololu_lib` because it diverges from upstream — the rename signals
that fact so future maintainers don't assume parity.

## Modifications relative to upstream

### `imu.py` — IMU I2C pin remap
The IMU's I2C bus was moved from GP4/GP5 (SDA/SCL) to GP19/GP20.
GP4/GP5 are now used for UART communication with the ESP Wi-Fi module;
GP19/GP20 were free in this build because line sensors are not used.
The original line is preserved in the file as a comment for reference.

### `buzzer.py` — lazy PWM acquire/release on GP7
Upstream attached the PWM peripheral to GP7 at module import time and
held the pin for the program's lifetime. GP7 is also our sonar recv1
trigger line, so this collision meant the recv1 trigger could not be
held low during normal operation, and the `Sonar()` constructor had to
be deferred until after the buzzer was done being used.

The patched version acquires PWM lazily on first use and releases it
after each tune (and at `Buzzer.off()` / construction), driving GP7
back to a clean low state via `Pin(7, Pin.OUT, value=0)`. The buzzer
still works normally; it just no longer holds the pin between beeps.

This is paired with a top-level `boot.py` that drives GP7, GP22, and
GP24 low at the earliest moment user code can run, to suppress MB1360
free-run mode during the MicroPython + Wi-Fi boot window.
