import ipaddress
import math
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import time
import string
import numpy as np

def profiledist2sonarsample(dists, samples):
    if len(dists) != 2 or len(samples) != 2:raise ValueError("x and y must each contain exactly 2 values")
    x1, x2 = dists
    y1, y2 = samples
    # Calculate the slope (m)
    m = (y2 - y1) / (x2 - x1)
    # Calculate the y-intercept (b)
    b = y1 - m * x1
    return b, m



def wrap_angle(angle, mode='deg180'):
    """
    Wrap angles to [-180, 180) or [0, 360).

    Parameters
    ----------
    angle : scalar, list/tuple, or numpy array
    mode : 'deg180' for [-180,180), 'deg360' for [0,360)

    Returns
    -------
    scalar, list, or numpy array
        Wrapped angle(s), matching input type.
    """
    # Track original type
    is_scalar = np.isscalar(angle)
    is_listlike = isinstance(angle, (list, tuple))
    # Convert to array for math
    arr = np.asarray(angle, dtype=float)
    # Wrap
    if mode == 'deg180':
        wrapped = ((arr + 180) % 360) - 180
    elif mode == 'deg360':
        wrapped = arr % 360
    else:
        raise ValueError("mode must be 'deg180' or 'deg360'")
    # Restore original type
    if is_scalar:
        return float(wrapped)
    if is_listlike:
        return wrapped.tolist()
    return wrapped

def sleep_ms(min_ms, max_ms=None):
    if max_ms is None:max_ms = min_ms
    ms = np.random.randint(min_ms, max_ms + 1)
    time.sleep(ms / 1000.0)

def make_code(n=8, prefix=""):
    # Use current time in microseconds
    ts = int(time.time() * 1e6)
    alphabet = string.ascii_uppercase  # only letters
    base = len(alphabet)
    # Convert timestamp into a base-26 string
    chars = []
    while ts > 0:
        ts, rem = divmod(ts, base)
        chars.append(alphabet[rem])
    code = ''.join(reversed(chars))
    # Pad or trim to desired length
    code = (code[-n:]).rjust(n, 'A')
    return f"{prefix}{code}"

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


def distance2samples(sample_rate, distance_m):
    speed_of_sound = 343.0
    t = 2.0 * distance_m / speed_of_sound  # seconds (round trip)
    n = int(round(t * float(sample_rate)))  # samples
    return n

def samples2distance(sample_rate, n_samples):
    speed_of_sound = 343.0
    t = float(n_samples) / float(sample_rate)
    distance_m = 0.5 * speed_of_sound * t
    return distance_m


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


def none2nan(array):
    arr = np.asarray(array, dtype=float)
    arr[arr == None] = np.nan  # noqa: E711
    return arr


def nonzero_indices(array):
    arr = np.asarray(array)
    indices = np.nonzero(arr)[0]
    return indices


def polar_plot(angle_deg, distance):
    min_distance = np.min(distance)
    if min_distance < 0: distance = distance - np.min(distance)
    radians = np.deg2rad(angle_deg)
    plt.axes(projection='polar')
    plt.polar(radians, distance)




def fill_nans_linear(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    isnan = np.isnan(x)
    if not isnan.any(): return x
    idx = np.arange(n)
    x[isnan] = np.interp(idx[isnan], idx[~isnan], x[~isnan])
    return x


def chip1d(array, n):
    length = len(array)
    if n < 1: return array
    if not length % 2 == 1: raise ValueError("array must have odd length")
    if 2 * n >= length: return array[length // 2]
    return array[n:-n]


def chip2d(array, n):
    width = array.shape[1]
    if n < 1: return array
    if not width % 2 == 1: raise ValueError("array must have odd width")
    if 2 * n >= width: return array[:, width // 2]
    return array[:, n:-n]

def chip(array, n):
    if array.ndim == 1: return chip1d(array, n)
    if array.ndim == 2: return chip2d(array, n)
    raise ValueError("array of wrong shape")

def get_extrema_positions(array, extrema='max'):
    if array.ndim == 1:
        if extrema == 'min': return np.nanargmin(array)
        if extrema == 'max': return np.nanargmax(array)
    if array.ndim == 2:
        if extrema == 'min': return np.nanargmin(array, axis=1)
        if extrema == 'max': return np.nanargmax(array, axis=1)

def get_side_average(array):
    # returns +1 if closest distances at right
    # return - 1 if closest distances at left
    if array.ndim == 1:
        length = len(array)
        middle = length // 2
        left_side_of_robot = np.nansum(array[middle + 1:])              #from middle + 1 to end
        right_side_of_robot = np.nansum(array[:middle])                 #from start to middle - 1
        side = np.sign(left_side_of_robot - right_side_of_robot)        #left farther => positive or obstacle at RIGHT
        return side
    if array.ndim == 2:
        width = array.shape[1]
        middle = width // 2
        left_side_of_robot = array[:, middle + 1:]  #from middle + 1 to end
        right_side_of_robot = array[:, :middle]     #from start to middle - 1
        left_side_of_robot = np.nansum(left_side_of_robot, axis=1)
        right_side_of_robot = np.nansum(right_side_of_robot, axis=1)
        side = np.sign(left_side_of_robot - right_side_of_robot)
        return side

def get_extrema_values(array, extrema='max'):
    if array.ndim == 1:
        if extrema == 'min': return np.nanmin(array)
        if extrema == 'max': return np.nanmax(array)
    if array.ndim == 2:
        if extrema == 'min': return np.nanmin(array, axis=1)
        if extrema == 'max': return np.nanmax(array, axis=1)


def plot_robot_positions(x, y, yaws_deg, dot_color='black', arrow_color='black', arrow_length=100, dot_cmap='viridis', arrow_cmap='viridis'):
    """
    Plot robot positions with xy positions as dots and yaw orientations as arrows.
    
    Supports scalar inputs for single robot positions as well as arrays for multiple robots.
    Both dot_color and arrow_color support arbitrary numerical arrays that are automatically
    mapped to the viridis colormap for visualization.
    
    Parameters
    ----------
    x : scalar, list, or numpy array
        X coordinate(s) of robot position(s). Can be a single scalar for one robot,
        or an array for multiple robots.
    y : scalar, list, or numpy array
        Y coordinate(s) of robot position(s). Must match x in length.
    yaws_deg : scalar, list, or numpy array
        Yaw orientation(s) in degrees (0 = right, 90 = up, 180 = left, 270 = down).
        Must match x and y in length.
    dot_color : str or array-like, optional
        Color for position dots (default: 'blue'). Supports:
        - Single color string (e.g., 'red', '#FF0000')
        - List of color strings for individual robots
        - Numerical array (values will be mapped to viridis colormap)
        - Single numerical value (applied to all robots)
    arrow_color : str or array-like, optional
        Color for orientation arrows (default: 'red'). Same options as dot_color.
    arrow_length : float, optional
        Length of arrows in data units (default: 0.5)
    dot_cmap : str, optional
        Colormap name for dot colors when using numerical arrays (default: 'viridis')
        Common options: 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                       'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter'
    arrow_cmap : str, optional
        Colormap name for arrow colors when using numerical arrays (default: 'viridis')
        Same options as dot_cmap.
    
    Returns
    -------
    matplotlib figure and axis objects
    
    Notes
    -----
    - When numerical arrays are provided for dot_color or arrow_color, colorbars 
      are automatically added to the plot to show the color scale.
    - Colorbars are only added when there is meaningful variation in the data 
      (i.e., not all values are identical).
    - String colors do not generate colorbars.
    - The figure layout is automatically adjusted when colorbars are added.
    
    Examples
    --------
    Single robot (scalar inputs):
        plot_robot_positions(1.0, 2.0, 45, dot_color='red')
    
    Multiple robots with string colors:
        plot_robot_positions([0, 1, 2], [0, 1, 0], [0, 90, 180],
                            dot_color=['red', 'blue', 'green'])
    
    Multiple robots with numerical color mapping:
        speeds = [1.2, 2.5, 0.8, 3.1]  # Speed values
        plot_robot_positions([0, 1, 2, 3], [0, 1, 0, -1], [0, 45, 90, 135],
                            dot_color=speeds, arrow_color='blue')
    
    Mixed color types:
        plot_robot_positions([0, 1, 2], [0, 1, 0], [0, 90, 180],
                            dot_color=['red', 'blue', 'green'],
                            arrow_color=[10, 20, 30])
    
    Custom colormaps:
        speeds = [1.2, 2.5, 0.8, 3.1]
        temperatures = [20, 25, 30, 35]
        plot_robot_positions([0, 1, 2, 3], [0, 1, 0, -1], [0, 45, 90, 135],
                            dot_color=speeds, dot_cmap='plasma',
                            arrow_color=temperatures, arrow_cmap='cool')
        # This will automatically add colorbars for both dot and arrow colors
    
    Colorbar example:
        # Numerical arrays automatically get colorbars with proper scaling
        battery_levels = [85, 60, 95, 40, 70]  # Percentage
        plot_robot_positions([0, 1, 2, 3, 4], [0, 1, 2, 1, 0], [0, 45, 90, 135, 180],
                            dot_color=battery_levels, dot_cmap='viridis')
        # A colorbar labeled 'Dot Color Value' will be added automatically
    """
    # Convert inputs to numpy arrays for consistent handling
    x = np.asarray(x)
    y = np.asarray(y)
    yaws_deg = np.asarray(yaws_deg)
    
    # Ensure we have at least 1D arrays for consistent handling
    if x.ndim == 0:  # scalar input
        x = x.reshape(1)
        y = y.reshape(1)
        yaws_deg = yaws_deg.reshape(1)
    
    # Create figure and axis
    fig = plt.gcf()
    ax = plt.gca()
    
    # Plot position dots with proper color handling
    if np.isscalar(dot_color):
        # Single color value (could be string like 'red' or numerical like 5.0)
        if isinstance(dot_color, str):
            # String color - use directly
            ax.scatter(x, y, color=dot_color, s=20, label='Robot Position')
        else:
            # Numerical value - create array of same value with colormap
            ax.scatter(x, y, c=np.full(len(x), dot_color), cmap=dot_cmap, s=50, label='Robot Position')
    else:
        dot_color_array = np.asarray(dot_color)
        if all(isinstance(color, str) for color in dot_color):
            # List of string colors
            ax.scatter(x, y, c=dot_color, s=20, label='Robot Position')
        else:
            # Numerical array - use specified colormap
            ax.scatter(x, y, c=dot_color_array, cmap=dot_cmap, s=20, label='Robot Position')
    
    # Plot orientation arrows
    if np.isscalar(arrow_color):
        arrow_color = [arrow_color] * len(x)
    else:
        arrow_color = np.asarray(arrow_color)
    
    # Convert yaw from degrees to radians for arrow plotting
    yaws_rad = np.deg2rad(yaws_deg)
    
    # Calculate arrow endpoints
    dx = arrow_length * np.cos(yaws_rad)
    dy = arrow_length * np.sin(yaws_rad)
    
    # Handle numerical arrow colors by creating a proper colormap
    if not np.isscalar(arrow_color) and not all(isinstance(color, str) for color in arrow_color):
        # Convert numerical array to normalized colors using specified colormap
        arrow_color = np.asarray(arrow_color)
        if len(arrow_color) > 1:
            # Normalize to [0, 1] range for colormap
            color_min, color_max = np.min(arrow_color), np.max(arrow_color)
            if color_max > color_min:
                normalized_values = (arrow_color - color_min) / (color_max - color_min)
            else:
                # All values are the same - use midpoint
                normalized_values = np.full(len(arrow_color), 0.5)
            arrow_colors_mapped = plt.cm.get_cmap(arrow_cmap)(normalized_values)
        else:
            # Single value - use midpoint of colormap
            arrow_colors_mapped = [plt.cm.get_cmap(arrow_cmap)(0.5)]
    else:
        # String colors - use as-is
        if np.isscalar(arrow_color):
            arrow_colors_mapped = [arrow_color] * len(x)
        else:
            arrow_colors_mapped = arrow_color
    
    for i in range(len(x)):
        ax.arrow(x[i], y[i], dx[i], dy[i], 
                 head_width=0.1, head_length=0.2, 
                 fc=arrow_colors_mapped[i], ec=arrow_colors_mapped[i],
                 length_includes_head=True, alpha=0.8)
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Robot Positions and Orientations')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    #ax.legend()
    
    # Add colorbars for numerical color arrays
    # We need to track if we added any colorbars to adjust the figure layout
    colorbars_added = []
    
    # Check if dot_color is a numerical array (not string colors)
    if not np.isscalar(dot_color) and len(dot_color) > 1:
        dot_color_array = np.asarray(dot_color)
        if not all(isinstance(color, str) for color in dot_color):
            # Numerical array with variation - add colorbar
            if np.max(dot_color_array) > np.min(dot_color_array):
                # Create a scalar mappable for the colorbar
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                norm = Normalize(vmin=np.min(dot_color_array), vmax=np.max(dot_color_array))
                sm = ScalarMappable(cmap=dot_cmap, norm=norm)
                sm.set_array([])  # Empty array for the scalar mappable
                
                # Add colorbar
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label('Dot Color Value')
                colorbars_added.append(cbar)
    
    # Check if arrow_color is a numerical array (not string colors)
    if not np.isscalar(arrow_color) and len(arrow_color) > 1:
        arrow_color_array = np.asarray(arrow_color)
        if not all(isinstance(color, str) for color in arrow_color):
            # Numerical array with variation - add colorbar
            if np.max(arrow_color_array) > np.min(arrow_color_array):
                # Create a scalar mappable for the colorbar
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                norm = Normalize(vmin=np.min(arrow_color_array), vmax=np.max(arrow_color_array))
                sm = ScalarMappable(cmap=arrow_cmap, norm=norm)
                sm.set_array([])  # Empty array for the scalar mappable
                
                # Add colorbar
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label('Arrow Color Value')
                colorbars_added.append(cbar)
    
    # Adjust figure layout if colorbars were added
    if colorbars_added:
        fig.tight_layout()
    
    return fig, ax

