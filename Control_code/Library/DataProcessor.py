import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from Library import Utils
from Library.DataStorage import DataReader


def interp_1d_index(a):
    a = np.asarray(a, dtype=float).copy()
    idx = np.arange(a.size)
    good = np.isfinite(a)
    if good.sum() == 0: return a
    a[~good] = np.interp(idx[~good], idx[good], a[good])
    return a


def interpolate_positions(rob_x, rob_y, rob_yaw_deg):
    rob_x = np.asarray(rob_x, dtype=float)
    rob_y = np.asarray(rob_y, dtype=float)
    missing = Utils.nonzero_indices(np.isnan(rob_x))
    rob_yaw_deg = np.asarray(rob_yaw_deg, dtype=float)
    # x, y: straight interpolation
    x_f = interp_1d_index(rob_x)
    y_f = interp_1d_index(rob_y)
    # yaw: circular interpolation
    yaw_rad = np.deg2rad(rob_yaw_deg)
    good = np.isfinite(yaw_rad)
    if good.sum() == 0:
        yaw_f_deg = rob_yaw_deg.astype(float).copy()
    else:
        s = np.sin(yaw_rad)
        c = np.cos(yaw_rad)

        s_f = interp_1d_index(s)
        c_f = interp_1d_index(c)

        yaw_f_rad = np.arctan2(s_f, c_f)
        yaw_f_deg = (np.rad2deg(yaw_f_rad) + 360.0) % 360.0

    return x_f, y_f, yaw_f_deg, missing


def load_mask_coordinates(data_reader):
    mask, meta, _ = load_env_mask(data_reader)
    x_coords, y_coords = mask2coordinates(mask, meta)
    return x_coords, y_coords


def mask2coordinates(mask, meta):
    min_x = meta["arena_bounds_mm"]["min_x"]
    max_y = meta["arena_bounds_mm"]["max_y"]
    mm_per_px = float(meta["map_mm_per_px"])
    mask = np.asarray(mask)
    rows, cols = np.nonzero(mask)
    x_coords = min_x + cols * mm_per_px + 0.5 * mm_per_px
    y_coords = max_y - rows * mm_per_px + 0.5 * mm_per_px
    return x_coords, y_coords


def is_data_reader(variable):
    reader_class = "<class 'Library.DataStorage.DataReader'>"
    return str(type(variable)) == reader_class


def get_env_dir(data_reader):
    base_folder = data_reader.base_folder if is_data_reader(data_reader) else str(data_reader)
    base_folder = Path(base_folder)
    if not base_folder.is_dir(): raise ValueError(f"{base_folder} is not a directory")
    # find folders starting with "env"
    env_dirs = sorted(p for p in base_folder.iterdir() if p.is_dir() and p.name.startswith("env"))
    if not env_dirs: raise FileNotFoundError("No folder starting with 'env' found")
    env_dir = env_dirs[0]
    return env_dir


def get_mask_file(data_reader):
    env_dir = get_env_dir(data_reader)
    mask_path = env_dir / "arena_mask_annotated.png"
    mask_path_fallback = env_dir / "mask.npy"
    if mask_path.is_file(): return mask_path, 'png'
    if mask_path_fallback.is_file(): return mask_path_fallback, 'npy'
    raise FileNotFoundError("No mask file found in env directory")


def read_image_mask(image_path, ref_rgb=(46, 194, 126), tol=35):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None: raise ValueError(f"Could not read image from {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    ref = np.array(ref_rgb, dtype=np.int16)
    dist = np.linalg.norm(img_rgb - ref, axis=2)
    mask = dist <= tol
    # optional cleanup
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8.astype(bool)


def read_npy_mask(npy_path):
    mask = np.load(npy_path)
    return mask


def load_env_mask(data_reader):
    mask = None
    base_folder = data_reader.base_folder if is_data_reader(data_reader) else str(data_reader)
    base_folder = Path(base_folder)
    if not base_folder.is_dir(): raise ValueError(f"{base_folder} is not a directory")
    env_dir = get_env_dir(data_reader)
    mask_path, ext = get_mask_file(data_reader)
    if ext == 'png': mask = read_image_mask(mask_path)
    if ext == 'npy': mask = read_npy_mask(mask_path)
    meta_path = env_dir / "meta.json"
    if not mask_path.is_file(): return False, False
    if not meta_path.is_file(): return False, False
    meta = json.load(open(meta_path))
    return mask, meta, ext


def world2robot(x_coords, y_coords, rob_x, rob_y, rob_yaw_deg):
    x_coords = np.asarray(x_coords, dtype=float)
    y_coords = np.asarray(y_coords, dtype=float)
    # translate into robot origin
    dx = x_coords - rob_x
    dy = y_coords - rob_y
    rob_yaw_rad = np.deg2rad(rob_yaw_deg)
    c = np.cos(rob_yaw_rad)
    s = np.sin(rob_yaw_rad)
    # rotate by -yaw (world -> robot)
    x_rel = c * dx + s * dy
    y_rel = -s * dx + c * dy
    return x_rel, y_rel


def read_robot_trajectory(data_reader):
    # This function keeps the yaw in degrees. We assume that we will convert as needed.
    rob_x = data_reader.get_field('position', 'x')
    rob_y = data_reader.get_field('position', 'y')
    rob_yaw_deg = data_reader.get_field('position', 'yaw')

    rob_x = np.array(rob_x)
    rob_y = np.array(rob_y)
    rob_yaw_deg = np.array(rob_yaw_deg)

    rob_x = Utils.none2nan(rob_x)
    rob_y = Utils.none2nan(rob_y)
    rob_yaw_deg = Utils.none2nan(rob_yaw_deg)

    rob_x, rob_y, rob_yaw_deg, missing = interpolate_positions(rob_x, rob_y, rob_yaw_deg)
    positions = {'rob_x': rob_x, 'rob_y': rob_y, 'rob_yaw_deg': rob_yaw_deg}
    positions = pd.DataFrame(positions)
    positions['missing'] = 0
    positions.loc[missing, 'missing'] = 1
    return positions


class DataProcessor:
    def __init__(self, data_reader):
        if type(data_reader) == str: data_reader = DataReader(data_reader)
        self.data_reader = data_reader
        self.mask, self.meta, self.ext = load_env_mask(data_reader)
        self.env_x, self.env_y = load_mask_coordinates(data_reader)
        self.positions = read_robot_trajectory(data_reader)
        self.filenames = self.data_reader.get_all_filenames()
        self.n = len(self.filenames)
        self.rob_x = self.positions.rob_x
        self.rob_y = self.positions.rob_y
        self.rob_yaw_deg = self.positions.rob_yaw_deg
        self.missing = self.positions.missing

    def plot_trajectory(self, show=True):
        arrow_len = 250
        rob_yaw_rad = np.deg2rad(self.rob_yaw_deg)
        cmap = ListedColormap(['blue', 'red'])
        u = np.cos(rob_yaw_rad) * arrow_len
        v = np.sin(rob_yaw_rad) * arrow_len

        plt.figure()
        plt.plot(self.rob_x, self.rob_y, '-', linewidth=2, alpha=0.5)
        plt.scatter(self.env_x, self.env_y, color='black', s=1, alpha=0.7)
        plt.scatter(self.rob_x, self.rob_y, c=self.missing, cmap=cmap, s=15)
        plt.quiver(self.rob_x, self.rob_y, u, v, angles='xy', scale_units='xy', scale=1, width=0.003)
        plt.axis('equal')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.grid(True)

        for i in range(0, self.n, 10): plt.text(self.rob_x[i], self.rob_y[i], str(i), color='green', fontsize=8)
        if show: plt.show()
        return plt.gcf()

    def plot_mask(self, show=True):
        min_x = self.meta["arena_bounds_mm"]["min_x"]
        max_x = self.meta["arena_bounds_mm"]["max_x"]
        min_y = self.meta["arena_bounds_mm"]["min_y"]
        max_y = self.meta["arena_bounds_mm"]["max_y"]
        extent = (min_x, max_x, min_y, max_y)

        plt.figure()
        plt.imshow(self.mask, extent=extent, origin='upper', cmap='gray')
        plt.title(f'Environment Mask ({self.ext})')
        plt.xlabel('x mm')
        plt.ylabel('y mm')
        if show: plt.show()


    def relative_environment_at(self, index, plot=False):
        selected_x = self.rob_x[index]
        selected_y = self.rob_y[index]
        selected_yaw_deg = self.rob_yaw_deg[index]
        rel_x, rel_y = world2robot(self.env_x, self.env_y, selected_x, selected_y, selected_yaw_deg)

        if plot:
            plt.figure()
            plt.scatter(rel_x, rel_y, s=1)
            plt.scatter(0, 0, c='red', marker='x', s=100, label='Robot Position')
            plt.arrow(0, 0, 100, 0, head_width=20, head_length=30, fc='red', ec='red', linewidth=2)
            plt.axis('equal')
            plt.title(f'Environment from robot frame at step {index}')
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.grid(True)
            plt.show()

        return rel_x, rel_y

    def get_data_at(self, index):
        data = self.data_reader.get_data_at(index)
        return data

    def get_sonar_data_at(self, index):
        data = self.data_reader.get_data_at(index)
        sonar_data = data['sonar_package']['sonar_data']
        configuration = data['sonar_package']['configuration']
        left_channel = configuration.left_channel
        right_channel = configuration.right_channel
        sonar_data = sonar_data[:, [left_channel, right_channel]]
        return sonar_data

    def print_data_overview(self):
        self.data_reader.print_data_overview()

    def environment_scan_at(self, index, min_az_deg, max_az_deg, n):
        # positive azimuth is to the left (CCW)
        # negative azimuth is to the right (CW)
        if n <= 0: raise ValueError("n must be a positive integer")
        if max_az_deg <= min_az_deg: raise ValueError("max_az_deg must be greater than min_az_deg")
        rel_x, rel_y = self.relative_environment_at(index, plot=False)
        rel_x = np.asarray(rel_x, dtype=float)
        rel_y = np.asarray(rel_y, dtype=float)
        angles_deg = np.rad2deg(np.arctan2(rel_y, rel_x))
        distances = np.hypot(rel_x, rel_y)

        edges = np.linspace(min_az_deg, max_az_deg, n + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        min_distances = np.full(n, np.nan, dtype=float)

        for i in range(n):
            in_slice = (angles_deg >= edges[i]) & (angles_deg < edges[i + 1])
            if np.any(in_slice): min_distances[i] = np.min(distances[in_slice])

        return centers, min_distances
