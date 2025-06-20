import numpy as np
from matplotlib import pyplot as plt

def sonar_plot(data, distance_axis, color=None, title='', yrange=None):
    x_max = float(distance_axis[-1])
    y_max = np.nanmax(data)
    if not np.isfinite(y_max): y_max = 1.0
    y_max = min(y_max, 50000)
    y_min = np.min(data)
    if yrange is not None:
        y_max = yrange[1]
        y_min = yrange[0]
    if color is None: color = 'black'
    # Horizontal axis = distance (x)
    ax = plt.gca()
    ax.plot(distance_axis, data)
    ax.set_xticks(np.arange(0, x_max + 0.05, 0.05), minor=True)
    ax.set_xticks(np.arange(0, x_max + 0.25, 0.25))
    ax.set_xlim(left=0, right=x_max)  # clamp to data range

    # Vertical axis = signal values (y)
    ax.set_yticks(np.arange(y_min, y_max, 500), minor=True)
    ax.set_yticks(np.arange(y_min, y_max, 5000))
    ax.set_ylim(bottom=y_min, top=y_max)

    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    ax.grid(which='major', color='darkgray', linestyle='--', linewidth=0.8)

    plt.xlabel('Distance [m]')
    plt.ylabel('Value [Arbitrary]')
    plt.title(title)



def shift_right(arr, n):
    if n <= 0:
        return arr.copy()
    shifted = np.empty_like(arr)
    shifted[:n] = arr[0]
    shifted[n:] = arr[:-n]
    return shifted

def trace_back(average, max_index=None, right=0, up=0):
    if max_index is not None: average[max_index:] = np.min(average)
    samples = len(average)
    flipped = np.flip(average)
    threshold = np.zeros(samples)
    current_max = 0
    for i in range(samples):
        if flipped[i] > current_max: current_max = flipped[i]
        threshold[i] = current_max
    threshold = np.flip(threshold)
    threshold = shift_right(threshold, right)
    threshold += up
    return threshold

def get_distance_axis(rate, samples):
    max_d = (343 / 2) * (samples / rate)
    distance_axis = np.linspace(0, max_d, samples)
    return distance_axis
