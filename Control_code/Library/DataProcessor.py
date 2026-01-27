import json
import functools
import numpy
from tqdm import tqdm
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from Library import Utils
from Library.DataStorage import DataReader
from Library import Guidance
from Library import Vectors



def collect(collation_results, field):
    collection = []
    for collated in collation_results: collection.append(collated[field])
    data = np.concatenate(collection, axis=0)
    data = np.asarray(data, dtype=np.float32)
    return data


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

def read_wall_mask(image_path, ref_rgb=(46, 194, 126), tol=35):
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    ref = np.array(ref_rgb, dtype=np.int16)
    dist = np.linalg.norm(img_rgb - ref, axis=2)
    mask = dist <= tol
    # optional cleanup
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8.astype(bool)


def read_path_mask( image_path, ref_rgb=(220, 40, 40),  tol=80):                 # distance threshold in RGB space
    min_area=20
    max_area=2000
    open_ksize=3
    close_ksize=5          # morphology to fill small gaps in dots

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None: raise ValueError(f"Could not read image from {image_path}")
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    ref = np.array(ref_rgb, dtype=np.int16)
    # RGB-distance threshold
    dist = np.linalg.norm(img_rgb - ref, axis=2)
    mask = dist <= tol
    # Morphological cleanup
    mask_u8 = (mask.astype(np.uint8) * 255)
    if open_ksize and open_ksize > 1:
        k = np.ones((open_ksize, open_ksize), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k)
    if close_ksize and close_ksize > 1:
        k = np.ones((close_ksize, close_ksize), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    # Output mask: only dot centers
    center_mask = np.zeros((h, w), dtype=bool)
    for lbl in range(1, num_labels):  # skip background
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area: continue
        if max_area is not None and area > max_area: continue
        cx, cy = centroids[lbl]
        ix = int(round(cx))
        iy = int(round(cy))
        if 0 <= ix < w and 0 <= iy < h: center_mask[iy, ix] = True  # row=y, col=x
    return center_mask


def read_npy_mask(npy_path):
    mask = np.load(npy_path)
    return mask


def load_arena_masks(data_reader):
    base_folder = data_reader.base_folder if is_data_reader(data_reader) else str(data_reader)
    base_folder = Path(base_folder)
    if not base_folder.is_dir(): raise ValueError(f"{base_folder} is not a directory")
    env_dir = get_env_dir(data_reader)
    annotation_path = env_dir / "arena_annotated.png"
    wall_mask = read_wall_mask(annotation_path)
    path_mask= read_path_mask(annotation_path)
    meta_path = env_dir / "meta.json"
    meta = json.load(open(meta_path))
    return wall_mask, path_mask, meta


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
    rob_yaw_deg = data_reader.get_field('position', 'yaw_deg')

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
        self.session = data_reader.base_folder
        self.data_reader = data_reader
        self.env_dir = get_env_dir(data_reader)
        self.wall_mask, self.path_mask, self.meta = None, None, None
        self.wall_x, self.wall_y = None, None
        self.path_x, self.path_y = None, None
        self.load_arena()

        self.positions = read_robot_trajectory(data_reader)
        self.filenames = self.data_reader.get_all_filenames()
        self.n = len(self.filenames)
        self.rob_x = self.positions.rob_x
        self.rob_y = self.positions.rob_y
        self.rob_yaw_deg = self.positions.rob_yaw_deg
        self.missing = self.positions.missing

    def print_data_overview(self):
        print('DATA IN DATA READER:')
        self.data_reader.print_data_overview()

    def get_motion(self):
        distances = self.data_reader.get_field('motion', 'distance')
        rotations = self.data_reader.get_field('motion', 'rotation')
        distances = np.asarray(distances)
        rotations = np.asarray(rotations)
        return distances, rotations

    def get_field(self, *fields):
        values = self.data_reader.get_field(*fields)
        values = np.asarray(values)
        return values

    def copy_annotated(self, original_path):
        dest_path = self.env_dir / "arena_mask_annotated.png"
        shutil.copy(original_path, dest_path)
        self.load_arena()

    def load_arena(self):
        try:
            self.wall_mask, self.path_mask, self.meta = load_arena_masks(self.data_reader)
            self.wall_x, self.wall_y = mask2coordinates(self.wall_mask, self.meta)
            self.path_x, self.path_y = mask2coordinates(self.path_mask, self.meta)
        except FileNotFoundError:
            print('Could not find the arena file')

    def plot_trajectory(self, show=True):
        arrow_len = 50
        rob_yaw_rad = np.deg2rad(self.rob_yaw_deg)
        cmap = ListedColormap(['blue', 'black'])
        u = np.cos(rob_yaw_rad) * arrow_len
        v = np.sin(rob_yaw_rad) * arrow_len

        plt.figure()
        plt.plot(self.rob_x, self.rob_y, '-', linewidth=2, alpha=0.5)
        plt.scatter(self.wall_x, self.wall_y, color='green', s=1, alpha=0.7)
        plt.scatter(self.path_x, self.path_y, color='red', s=2, alpha=0.7)
        plt.scatter(self.rob_x, self.rob_y, c=self.missing, cmap=cmap, s=15)
        plt.quiver(self.rob_x, self.rob_y, u, v, angles='xy', scale_units='xy', scale=1, width=0.003)
        plt.axis('equal')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.grid(True)

        for i in range(0, self.n, 10): plt.text(self.rob_x[i], self.rob_y[i], str(i), color='green', fontsize=8)
        if show: plt.show()
        return plt.gcf()

    def get_arena_extent(self):
        min_x = self.meta["arena_bounds_mm"]["min_x"]
        max_x = self.meta["arena_bounds_mm"]["max_x"]
        min_y = self.meta["arena_bounds_mm"]["min_y"]
        max_y = self.meta["arena_bounds_mm"]["max_y"]
        extent = (min_x, max_x, min_y, max_y)
        return extent


    def plot_arena(self, show=True):
        plt.figure()
        plt.scatter(self.wall_x, self.wall_y, color='green', s=1)
        plt.scatter(self.path_x, self.path_y, color='red', s=5)
        plt.scatter(self.rob_x, self.rob_y, color='black', s=1)
        plt.xlabel('x mm')
        plt.ylabel('y mm')
        plt.axis('equal')
        if show: plt.show()


    def relative_wall(self, rob_x, rob_y, rob_yaw_deg, plot=False):
        rel_x, rel_y = world2robot(self.wall_x, self.wall_y, rob_x, rob_y, rob_yaw_deg)
        if plot:
            plt.figure()
            plt.scatter(rel_x, rel_y, s=1)
            plt.scatter(0, 0, c='red', marker='x', s=100, label='Robot Position')
            plt.arrow(0, 0, 100, 0, head_width=20, head_length=30, fc='red', ec='red', linewidth=2)
            plt.axis('equal')
            plt.title('Environment from robot frame')
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.grid(True)
            plt.show()
        return rel_x, rel_y


    def relative_wall_at(self, index, plot=False):
        selected_x = self.rob_x[index]
        selected_y = self.rob_y[index]
        selected_yaw_deg = self.rob_yaw_deg[index]
        rel_x, rel_y = self.relative_wall(selected_x, selected_y, selected_yaw_deg, plot=plot)
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

    def get_sonar_data(self, flatten=False):
        data = self.data_reader.get_data_at(0)
        configuration = data['sonar_package']['configuration']
        left_channel = configuration.left_channel
        right_channel = configuration.right_channel
        sonar_data = self.get_field('sonar_package', 'sonar_data')
        sonar_data = np.array(sonar_data, dtype=np.float32)
        sonar_data = sonar_data[:, :, [left_channel, right_channel]]
        if flatten: sonar_data = sonar_data.reshape(sonar_data.shape[0], -1)
        return sonar_data

    def wall_scan(self, rob_x, rob_y, rob_yaw_deg, min_az_deg, max_az_deg, n):
        selected_x = rob_x
        selected_y = rob_y
        selected_yaw_deg = rob_yaw_deg
        rel_x, rel_y = self.relative_wall(selected_x, selected_y, selected_yaw_deg, plot=False)
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

    def wall_scan_at(self, index, min_az_deg, max_az_deg, n):
        selected_x = self.rob_x[index]
        selected_y = self.rob_y[index]
        selected_yaw_deg = self.rob_yaw_deg[index]
        centers, profile = self.wall_scan(selected_x, selected_y, selected_yaw_deg, min_az_deg, max_az_deg, n)
        return centers, profile

    def get_position_at(self, index):
        selected_x = self.rob_x[index]
        selected_y = self.rob_y[index]
        selected_yaw_deg = self.rob_yaw_deg[index]
        return selected_x, selected_y, selected_yaw_deg

    @functools.cache
    def collate_data(self, az_min=-45, az_max=45, az_steps=None,d=150):
        prepared_data = {}
        sonar_block = self.get_sonar_data(flatten=False)
        sonar = self.get_sonar_data(flatten=True)
        corrected_iid = self.get_field('sonar_package', 'corrected_iid')
        corrected_distance = self.get_field('sonar_package', 'corrected_distance')
        distances, rotations = self.get_motion()
        rob_x = self.rob_x
        rob_y = self.rob_y
        rob_yaw_deg = self.rob_yaw_deg
        vector_field, polyline = self.get_vector_field(d=d)

        prepared_data['rob_x'] = rob_x
        prepared_data['rob_y'] = rob_y
        prepared_data['rob_yaw_deg'] = rob_yaw_deg
        prepared_data['sonar_block'] = sonar_block
        prepared_data['sonar_data'] = sonar
        prepared_data['rotations'] = rotations
        prepared_data['distance'] = distances
        prepared_data['vector_field'] = vector_field
        prepared_data['polyline'] = polyline
        prepared_data['corrected_iid'] = corrected_iid
        prepared_data['corrected_distance'] = corrected_distance
        if az_steps is None: return prepared_data
        # Get profiles
        profiles = []
        max_directions = []

        for index in tqdm(range(self.n)):
            centers, profile = self.wall_scan_at(index, az_min, az_max, az_steps)
            profile = Utils.fill_nans_linear(profile)
            max_index = numpy.argmax(profile)
            profile = np.asarray(profile, dtype=np.float32)
            profiles.append(profile)
            max_directions.append(profile[max_index])
        prepared_data['profiles'] = np.asarray(profiles, dtype=np.float32)
        prepared_data['centers'] = np.asarray(centers, dtype=np.float32)
        prepared_data['max_directions'] = np.asarray(max_directions, dtype=np.float32)
        return prepared_data

    def get_vector_field(self, d=100, plot=False):
        polyline = Guidance.build_polyline_from_xy(self.path_x, self.path_y, plot=plot)
        rob_x = self.rob_x
        rob_y = self.rob_y
        rob_yaw_deg = self.rob_yaw_deg
        df = Guidance.guidance_vector_field_batch(rob_x, rob_y, rob_yaw_deg, polyline, d, visualize=plot)
        return df, polyline

    def get_wall_vector_field(
        self,
        plot=False,
        close_iter=2,
        dilate_iter=2,
        seed_xy=None,
    ):
        rob_x = self.rob_x
        rob_y = self.rob_y
        rob_yaw_deg = self.rob_yaw_deg
        if seed_xy is None:
            seed_xy = (
                float(rob_x[0]),
                float(rob_y[0]),
            )
        seed_col, seed_row = Vectors.world_to_pixel(seed_xy[0], seed_xy[1], self.meta)
        df, dist, inside, mask = Vectors.wall_vector_field_batch(
            rob_x,
            rob_y,
            rob_yaw_deg,
            self.wall_mask,
            self.meta,
            seed_xy=(seed_col, seed_row),
            close_iter=close_iter,
            dilate_iter=dilate_iter,
            visualize=plot,
            invert_y=False,
            title="Wall repulsion field (poses)",
        )
        return df, dist, inside, mask
