from pololu_3pi_2040_robot import robot
import time

class LEDs:
    COLORS = {
        "off":       [0, 0, 0],
        "black":     [0, 0, 0],
        "red":       [255, 0, 0],
        "green":     [0, 255, 0],
        "blue":      [0, 0, 255],
        "yellow":    [255, 255, 0],
        "cyan":      [0, 255, 255],
        "magenta":   [255, 0, 255],
        "white":     [255, 255, 255],
        "orange":    [255, 128, 0],
        "purple":    [128, 0, 255],
        "pink":      [255, 105, 180],
        "dim_red":   [64, 0, 0],
        "dim_blue":  [0, 0, 64],
        "dim_green": [0, 64, 0],
    }

    COLORMAPS = {
        "blue_red":         ["blue", "red"],
        "blue_green_red":   ["blue", "green", "red"],
        "grayscale":        ["black", "white"],
        "hot":              ["black", "red", "orange", "yellow", "white"],
        "cool":             ["cyan", "blue", "purple"],
        "spring":           ["magenta", "yellow"],
        "autumn":           ["red", "orange", "yellow"],
        "viridis": [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
    }

    def __init__(self):
        self.leds = robot.RGBLEDs()
        self.num_leds = 6
        self.colors = [[0, 0, 0]] * self.num_leds
        self.leds.set_brightness(5)

    def set_brightness(self, level):
        """Set global brightness (0–31)."""
        self.leds.set_brightness(level)

    def set(self, index, color):
        """Set a single LED by index to a color (name or [R,G,B]), then show."""
        rgb = self._resolve_color(color)
        if 0 <= index < self.num_leds:
            self.colors[index] = rgb
            self.leds.set(index, rgb)
            self.show()

    def set_all(self, color):
        """Set all LEDs to a color (name or [R,G,B]), then show."""
        rgb = self._resolve_color(color)
        for i in range(self.num_leds):
            self.colors[i] = rgb
            self.leds.set(i, rgb)
        self.show()

    def clear(self):
        """Turn off all LEDs."""
        self.set_all("off")

    def show(self):
        """Update LED hardware with stored colors."""
        self.leds.show()

    def flash(self, color, delay_ms=200):
        """Flash all LEDs briefly with a given color."""
        self.set_all(color)
        time.sleep_ms(delay_ms)
        self.clear()

    def set_gradient(self, index, value, domain=(0, 1), cmap="blue_red"):
        """
        Set LED color using interpolated color from a colormap.

        - index: which LED to set (0–5)
        - value: scalar input
        - domain: (min, max) range for mapping
        - cmap: name of colormap to use (must be in COLORMAPS)
        """
        colors = self.COLORMAPS.get(cmap, ["blue", "red"])
        vmin, vmax = domain
        if vmax == vmin:
            t = 0.0
        else:
            t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))

        n = len(colors) - 1
        segment = min(int(t * n), n - 1)
        local_t = (t * n) - segment

        c1 = self._resolve_color(colors[segment])
        c2 = self._resolve_color(colors[segment + 1])
        rgb = [int(c1[i] + (c2[i] - c1[i]) * local_t) for i in range(3)]

        self.set(index, rgb)

    def _resolve_color(self, color):
        """Convert a named or RGB color to RGB with green brightness scaled."""
        if isinstance(color, str):
            rgb = self.COLORS.get(color.lower(), [0, 0, 0])
        else:
            rgb = color
        return [rgb[0], rgb[1] // 3, rgb[2]]  # dim green for visual balance
