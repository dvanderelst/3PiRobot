import ipaddress
import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def make_ticks(vmin, vmax, steps, preferred=8):
    """Return (ticks, chosen_step) for [vmin, vmax] using the step from `steps`
    that gives a tick count closest to `preferred`."""
    if vmax < vmin: vmin, vmax = vmax, vmin
    span = vmax - vmin
    best = None
    best_ticks = None

    for step in steps:
        if step <= 0: continue
        start = math.floor(vmin / step) * step
        end   = math.ceil(vmax / step) * step
        # robust arange with a small epsilon to include endpoint
        ticks = np.arange(start, end + step * 0.5, step)
        # keep only visible ticks (optional, avoids ticks outside limits)
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        n = len(ticks)
        score = abs(n - preferred)

        # tie-break: prefer slightly denser (larger n), then smaller step
        key = (score, -n, step)
        if (best is None) or (key < best[0]):
            best = (key, step)
            best_ticks = ticks

    return best_ticks if best_ticks is not None else np.array([]), (best[1] if best else None)


def find_closest_value_index(array, target):
    array = np.asarray(array)
    idx = (np.abs(array - target)).argmin()
    #value = array[idx]
    return idx

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

def sonar_plot(sonar_package, title='', yrange=None, color='black', label=None, distance_axis=None):
    sonar_data = sonar_package['sonar_data']
    if distance_axis is None: distance_axis = sonar_package['raw_distance_axis']

    x_min = float(distance_axis[0])
    x_max = float(distance_axis[-1])
    y_min = float(np.nanmin(sonar_data)) - 1000.0
    y_max = float(np.nanmax(sonar_data)) + 1000.0
    if yrange is not None: y_min, y_max = yrange

    ax = plt.gca()
    dim = sonar_data.shape
    labels_set = False

    if len(dim) == 2 and dim[1] == 3:
        ax.plot(distance_axis, sonar_data[:, 0], color='grey', marker='.', label='Emitter')
        ax.plot(distance_axis, sonar_data[:, 1], color='blue',  marker='.', label='Left Channel')
        ax.plot(distance_axis, sonar_data[:, 2], color='red',   marker='.', label='Right Channel')
        labels_set = True
    else:
        ax.plot(distance_axis, sonar_data, color=color, marker='.', label=label)

    # ---- Adaptive ticks for main x and y axes ----
    x_ticks, _ = make_ticks(x_min, x_max, steps=[0.025, 0.05, 0.1, 0.2, 0.5, 1], preferred=9)
    y_ticks, _ = make_ticks(y_min, y_max, steps=[500, 1000, 2000, 5000], preferred=8)

    if x_ticks.size: ax.set_xticks(x_ticks)
    ax.set_xlim(x_min, x_max)

    if y_ticks.size: ax.set_yticks(y_ticks)
    ax.set_ylim(y_min, y_max)
    # ---- Adaptive top axis using indices ----
    n = len(distance_axis)
    # Use indices 0..n-1 as the "range" and pick adaptive ticks
    index_ticks, _ = make_ticks(0, n-1, steps=[10, 20, 50, 100, 200], preferred=10)
    ax_top = ax.secondary_xaxis('top')
    # Convert index ticks into positions on the distance axis
    pos_ticks = distance_axis[index_ticks.astype(int)]
    ax_top.set_ticks(pos_ticks)
    ax_top.set_xticklabels([str(int(i)) for i in index_ticks])
    ax_top.set_xlabel("Index")
    ax.grid(True, which='both', axis='both')

    plt.xlabel('Raw Distance [m]')
    plt.ylabel('Value [Arbitrary]')
    if labels_set: plt.legend()
    plt.title(title)


def get_distance_axis(sample_rate, samples):
    speed_of_sound = 343.0
    n = np.arange(samples, dtype=float)            # 0, 1, ..., samples-1
    t = n / float(sample_rate)                     # seconds
    d = 0.5 * float(speed_of_sound) * t            # meters (round trip)
    return d

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


def draw_integration_box(ax, bounds, color='gray', alpha=0.3, onset_color='red'):
    if not 2 <= len(bounds) <= 4:
        raise ValueError("bounds must have 2 to 4 elements: (x_min, x_max[, y_min, y_max])")
    x_min, x_max = bounds[0], bounds[1]
    # Handle y bounds
    if len(bounds) >= 4:
        y_min, y_max = bounds[2], bounds[3]
    else:
        ymin, ymax = ax.get_ylim()
        y_min = bounds[2] if len(bounds) == 3 else ymin
        y_max = ymax
    # Draw shaded rectangle
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,  # width
        y_max - y_min,  # height
        linewidth=0,
        facecolor=color,
        alpha=alpha,
        label='Integration Extent'
    )
    ax.add_patch(rect)
    # Draw onset line
    ax.axvline(x_min, color=onset_color, linestyle='--', label='Onset')
