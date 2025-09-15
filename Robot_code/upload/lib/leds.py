from pololu_3pi_2040_robot import robot
import time

COLORS = {
    "off": [0, 0, 0],
    "black": [0, 0, 0],
    "red": [255, 0, 0],
    "green": [0, 255, 0],
    "blue": [0, 0, 255],
    "yellow": [255, 255, 0],
    "cyan": [0, 255, 255],
    "magenta": [255, 0, 255],
    "white": [255, 255, 255],
    "orange": [255, 128, 0],
    "purple": [128, 0, 255],
    "pink": [255, 105, 180],
    "dim_red": [64, 0, 0],
    "dim_blue": [0, 0, 64],
    "dim_green": [0, 64, 0],
}

COLORMAPS = {
    "blue_black": ["blue", "black"],
    "blue_red": ["blue", "red"],
    "blue_green_red": ["blue", "green", "red"],
    "grayscale": ["black", "white"],
    "hot": ["black", "red", "orange", "yellow", "white"],
    "cool": ["cyan", "blue", "purple"],
    "spring": ["magenta", "yellow"],
    "autumn": ["red", "orange", "yellow"],
    "viridis": [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ],
}

class LEDs:


    def __init__(self):
        self.leds = robot.RGBLEDs()
        self.num_leds = 6
        self.colors = [[0, 0, 0] for _ in range(self.num_leds)]
        self.toggle_colors = {}  # index -> {"idx": int, "palette": list[[R,G,B]]}
        self._batch_depth = 0
        self._dirty = False
        self.set_brightness(5)

    # ─────────────── Batch API (no contextlib) ───────────────
    def begin(self):
        """Start a batch: defer .show() until end()."""
        self._batch_depth += 1

    def end(self, show=True):
        """End a batch. When outmost ends and dirty, push to hardware."""
        if self._batch_depth > 0:
            self._batch_depth -= 1
        if self._batch_depth == 0 and self._dirty and show:
            self.show()

    def _mark_dirty(self):
        if self._batch_depth > 0:
            self._dirty = True

    # ─────────────── Basic controls ───────────────
    def set_brightness(self, level):
        self.leds.set_brightness(level)

    def set(self, index, color, show=True):
        """Set a single LED (0..num_leds-1) to a color."""
        if 0 <= index < self.num_leds:
            rgb = self.resolve_color(color)
            self.colors[index] = [rgb[0], rgb[1], rgb[2]]
            self.leds.set(index, self.colors[index])
            if self._batch_depth == 0 and show:
                self.show()
            else:
                self._mark_dirty()

    def set_many(self, pairs, show=True):
        """
        Set multiple LEDs in one call.
        pairs: iterable of (index, color)
        """
        self.begin()
        try:
            for index, color in pairs:
                self.set(index, color, show=False)
        finally:
            self.end(show=show)

    def set_all(self, color, show=True):
        """Set all LEDs to a color."""
        rgb = self.resolve_color(color)
        self.begin()
        try:
            for i in range(self.num_leds):
                self.colors[i] = [rgb[0], rgb[1], rgb[2]]
                self.leds.set(i, self.colors[i])
        finally:
            self.end(show=show)

    def clear(self, show=True):
        self.set_all("off", show=show)

    def show(self):
        """Push buffered LED state to hardware."""
        self._dirty = False
        self.leds.show()

    def flash(self, color, delay_ms=200):
        """Flash all LEDs briefly, then clear."""
        self.begin()
        try:
            self.set_all(color, show=False)
        finally:
            self.end(show=True)
        time.sleep_ms(delay_ms)
        self.clear()

    # ─────────────── Gradients / colormaps ───────────────
    def set_gradient(self, index, value, domain=(0, 1), cmap="blue_red", show=True):
        colors = COLORMAPS.get(cmap, ["blue", "red"])
        vmin, vmax = domain
        t = 0.0 if vmax == vmin else max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
        n = len(colors) - 1
        segment = min(int(t * n), n - 1)
        local_t = (t * n) - segment
        c1 = self.resolve_color(colors[segment])
        c2 = self.resolve_color(colors[segment + 1])
        rgb = [int(c1[i] + (c2[i] - c1[i]) * local_t) for i in range(3)]
        self.set(index, rgb, show=show)

    # ─────────────── Toggle system ───────────────
    def set_toggle_colors(self, index, colors, start_at=0, show=True):
        """Configure a per-LED palette for toggling and (optionally) show it."""
        if not (0 <= index < self.num_leds):
            return
        palette = [self.resolve_color(c) for c in colors]
        if not palette:
            return
        start_at = max(0, min(start_at, len(palette) - 1))
        self.toggle_colors[index] = {"idx": start_at, "palette": palette}
        self.set(index, palette[start_at], show=show)

    def toggle_color(self, index, step=1, show=True):
        """Advance LED's color by step (negative step = backward)."""
        state = self.toggle_colors.get(index)
        if not state:
            return
        n = len(state["palette"])
        state["idx"] = (state["idx"] + step) % n
        self.set(index, state["palette"][state["idx"]], show=show)

    def toggle_all_colors(self, step=1, indices=None, show=True):
        """Advance all configured toggles (or a subset)."""
        targets = indices if indices is not None else list(self.toggle_colors.keys())
        self.begin()
        try:
            for i in targets:
                self.toggle_color(i, step=step, show=False)
        finally:
            self.end(show=show)

    def clear_toggle(self, index):
        self.toggle_colors.pop(index, None)

    # ─────────────── Utilities ───────────────
    def resolve_color(self, color):
        """Map a name or [R,G,B] to an RGB list (green scaled for balance)."""
        if isinstance(color, str):
            base = COLORS.get(color.lower(), [0, 0, 0])
        else:
            base = color
        return [base[0], base[1] // 3, base[2]]

    def probe(self, dwell=0.6):
        """Light each index alone so you can see which LED is which."""
        for i in range(self.num_leds):
            self.begin()
            try:
                self.set_all("off", show=False)
                self.set(i, "white", show=False)
            finally:
                self.end(show=True)
            time.sleep(dwell)
        self.clear()
