import ipaddress
import numpy as np
from matplotlib import pyplot as plt


def is_valid_ip(s):
    if not isinstance(s, str):
        return False
    try:
        ipaddress.IPv4Address(s)
        return True
    except ipaddress.AddressValueError:
        return False

def compare_configurations(config1, config2):
        sample_rate1 = config1.sample_rate
        samples1 = config1.samples
        sample_rate2 = config2.sample_rate
        samples2 = config2.samples

        sample_rate_same = sample_rate1 == sample_rate2
        samples_same = samples1 == samples2
        matches = sample_rate_same and samples_same
        return matches

def sonar_plot(data, sample_rate, title='', yrange=None, color='black', label=None):
    samples = data.shape[0]
    distance_axis = get_distance_axis(sample_rate, samples)
    x_max = float(distance_axis[-1])
    y_max = np.nanmax(data)
    if not np.isfinite(y_max): y_max = 1.0
    y_max = min(y_max, 50000)
    y_min = np.min(data)
    if yrange is not None:
        y_max = yrange[1]
        y_min = yrange[0]
    # Horizontal axis = distance (x)
    ax = plt.gca()
    dim = data.shape
    labels_set = False
    # assumes the order of the channels is: [emitter, left, right]
    if len(dim) == 2 and dim[1] == 3:
        ax.plot(distance_axis, data[:, 0], color='black', marker='.', label='Emitter')
        ax.plot(distance_axis, data[:, 1], color='blue', marker='.' , label='Left Channel')
        ax.plot(distance_axis, data[:, 2], color='red', marker='.', label='Right Channel')
        labels_set = True
    # makes no assumption about the number of channels
    else:
        ax.plot(distance_axis, data, color=color, marker='.', label=label)

    ax.set_xticks(np.arange(0, x_max + 0.05, 0.05), minor=True)
    ax.set_xticks(np.arange(0, x_max + 0.25, 0.25))
    ax.set_xlim(left=0, right=x_max)  # clamp to data range

    # Vertical axis = signal values (y)
    ax.set_yticks(np.arange(y_min, y_max, 500), minor=True)
    ax.set_yticks(np.arange(y_min, y_max, 5000))
    ax.set_ylim(bottom=y_min, top=y_max)

    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    ax.grid(which='major', color='darkgray', linestyle='--', linewidth=0.8)

    # Add indices
    indices = np.arange(0, len(distance_axis), step=10)
    tick_positions = distance_axis[indices]

    ax_top = ax.secondary_xaxis('top')
    ax_top.set_ticks(tick_positions)
    ax_top.set_xticklabels([str(i) for i in indices])
    ax_top.set_xlabel("Index")

    plt.xlabel('Distance [m]')
    plt.ylabel('Value [Arbitrary]')
    if labels_set: plt.legend()
    plt.title(title)


def get_distance_axis(sample_rate, samples):
    max_d = (343 / 2) * (samples / sample_rate)
    distance_axis = np.linspace(0, max_d, samples)
    return distance_axis

def fit_linear_calibration(real_distance1, raw_distances1, real_distance2, raw_distances2):
    raw_distances1 = np.asarray(raw_distances1)
    raw_distances2 = np.asarray(raw_distances2)

    # Combine raw distances and real values
    x = np.concatenate([raw_distances1, raw_distances2])
    y = np.array([real_distance1] * len(raw_distances1) +
                 [real_distance2] * len(raw_distances2))

    # Compute slope and intercept using least squares
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    a = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    b = y_mean - a * x_mean

    return a, b
