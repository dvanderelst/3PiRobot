class RollingAverage:
    def __init__(self, size):
        self.size = size
        self.measurements = []
        self.total_count = 0

    def add(self, value):
        self.measurements.append(value)
        self.total_count += 1
        # Keep only the last n measurements
        if len(self.measurements) > self.size:
            self.measurements.pop(0)  # Remove oldest

    def average(self):
        if not self.measurements:
            return 0
        return sum(self.measurements) / len(self.measurements)

    def count(self):
        """Returns number of measurements currently stored (max = size)"""
        return len(self.measurements)

    def total_samples(self):
        """Returns total number of samples processed since creation/reset"""
        return self.total_count

    def reset_total_count(self):
        """Resets the total count of samples processed to zero"""
        self.total_count = 0

    def reset(self):
        """Clears all measurements and resets total count"""
        self.measurements = []
        self.total_count = 0