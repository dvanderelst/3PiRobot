import math

class Converter:
    def __init__(self):
        """Initialize with encoder counts per revolution and wheel diameter in mm."""
        counts_per_rev = 909.72
        wheel_diameter_mm = 31
        self.max_mps = 0.4 #This comes from the specs. Probably conservative.
        self.counts_per_rev = counts_per_rev
        self.wheel_diameter_m = wheel_diameter_mm / 1000  # convert to meters
        self.wheel_circumference_m = math.pi * self.wheel_diameter_m
        self.meters_per_count = self.wheel_circumference_m / self.counts_per_rev
        self.counts_per_meter = self.counts_per_rev / self.wheel_circumference_m
        self.max_cps = self.mps2cps(self.max_mps)

    def cps2mps(self, cps):
        """Convert encoder counts per second to meters per second."""
        if cps < 0: cps = 0
        if cps > self.max_cps: cps = self.max_cps
        return cps * self.meters_per_count

    def mps2cps(self, mps):
        """Convert meters per second to encoder counts per second."""
        if mps < 0: mps = 0
        if mps > self.max_mps: mps = self.max_mps
        return mps * self.counts_per_meter
