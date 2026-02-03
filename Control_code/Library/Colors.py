from matplotlib import pyplot as plt
import numpy as np

class ColorMap:
    def __init__(self, n=10):
        self.n = n
        self.cm = 'tab10'
        self.colors = []
        self.current_index = 0
        self.set_colors()


    def set_colors(self, cm=None, n=None):
        if cm is None: cm = self.cm
        if n is None: n = self.n
        self.cm = cm
        self.n = n
        cmap = plt.get_cmap(self.cm)
        self.colors = cmap(np.linspace(0, 1, n))

    def get_color(self, i):
        index = i % len(self.colors)
        return self.colors[index]

    def get_next_color(self):
        index = self.current_index % len(self.colors)
        self.current_index += 1
        return self.colors[index]
