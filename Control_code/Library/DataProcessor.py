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
from datetime import datetime


def find_bounding_box_across_views(views_stack, black_threshold=10):
    """
    Find the minimal bounding box that contains all non-black pixels across a stack of views.
    
    This helper function is useful for neural network preprocessing to identify
    the region of interest in conical views and crop out empty black borders.
    
    Parameters
    ----------
    views_stack : numpy.ndarray
        Stack of views with shape (N, H, W, C) where C=3 for RGB
    black_threshold : int, optional
        Threshold for considering a pixel as black (0-255).
        A pixel is considered black if all RGB channels are <= this threshold.
        Default: 10 (allows for some noise in black areas)
        
    Returns
    -------
    tuple or None
        Bounding box coordinates (x_min, y_min, x_max, y_max) as a tuple,
        or None if no non-black pixels are found
        
    Examples
    --------
    >>> views = np.random.randint(0, 256, (10, 128, 128, 3), dtype=np.uint8)
    >>> bbox = find_bounding_box_across_views(views)
    >>> if bbox:
    ...     x_min, y_min, x_max, y_max = bbox
    ...     cropped_views = views[:, y_min:y_max, x_min:x_max, :]
    """
    # Validate input
    if not isinstance(views_stack, np.ndarray) or views_stack.ndim != 4:
        raise ValueError(f"Expected 4D array (N, H, W, C), got shape: {views_stack.shape}")
    
    if views_stack.shape[3] != 3:
        raise ValueError(f"Expected 3 color channels (RGB), got: {views_stack.shape[3]}")
    
    # Create a mask where True indicates non-black pixels
    # A pixel is non-black if ANY channel exceeds the black threshold
    non_black_mask = np.any(views_stack > black_threshold, axis=3)
    
    # Find non-black pixels across the entire stack (collapse N dimension)
    any_non_black = np.any(non_black_mask, axis=0)  # Shape: (H, W)
    
    # Find bounding box coordinates
    non_black_rows, non_black_cols = np.where(any_non_black)
    
    if len(non_black_rows) == 0 or len(non_black_cols) == 0:
        # No non-black pixels found
        return None
    
    # Calculate bounding box
    y_min, y_max = np.min(non_black_rows), np.max(non_black_rows)
    x_min, x_max = np.min(non_black_cols), np.max(non_black_cols)
    
    # Add a small margin to be safe
    margin = 2
    y_min = max(0, y_min - margin)
    y_max = min(views_stack.shape[1], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(views_stack.shape[2], x_max + margin)
    
    return (x_min, y_min, x_max, y_max)
from matplotlib.colors import ListedColormap
import os

from Library import Utils
from Library.DataStorage import DataReader
import math


def _compute_curvature_single(args):
    i, az, dist, distance_cutoff_mm, cfg = args
    from Library.ProfileCurvature import profile2curvature
    dist = np.asarray(dist, dtype=np.float32).copy()
    invalid = (dist < 0.0) | (dist > float(distance_cutoff_mm))
    dist[invalid] = np.nan
    kappa = float(profile2curvature(az_deg=np.asarray(az, dtype=np.float32), dist_mm=dist, config=cfg, return_debug=False))
    return i, kappa



#from Library import Guidance
#from Library import Vectors


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
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
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
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    ref = np.array(ref_rgb, dtype=np.int16)
    dist = np.linalg.norm(img_rgb - ref, axis=2)
    mask = dist <= tol
    # optional cleanup
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return mask_u8.astype(bool)


def read_path_mask(image_path, ref_rgb=(220, 40, 40), tol=80):  # distance threshold in RGB space
    min_area = 20
    max_area = 2000
    open_ksize = 3
    close_ksize = 5  # morphology to fill small gaps in dots

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
    path_mask = read_path_mask(annotation_path)
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


def robot2world(az_deg, dist, rob_x, rob_y, rob_yaw_deg):
    """
    Convert robot-relative positions (azimuth and distance) to world coordinates.
    
    This is the inverse of world2robot function.
    
    Parameters
    ----------
    az_deg : scalar, list, or numpy array
        Azimuth angles in degrees (0 = forward, 90 = up/left, 180 = backward, 270 = down/right)
        relative to the robot's current orientation.
        
        IMPORTANT: Azimuths follow MATHEMATICAL convention (counter-clockwise):
        - 0° = Forward (positive X in robot frame)
        - 90° = Up/Left (positive Y in robot frame) 
        - 180° = Backward (negative X in robot frame)
        - 270° = Down/Right (negative Y in robot frame)
        
        This means azimuths increase COUNTER-CLOCKWISE from the forward direction.
    dist : scalar, list, or numpy array
        Distances from the robot in the same units as rob_x, rob_y
    rob_x, rob_y : scalar
        Robot's position in world coordinates
    rob_yaw_deg : scalar
        Robot's orientation in degrees (0 = right/east, 90 = up/north, etc.)
    
    Returns
    -------
    x_world, y_world : numpy arrays
        World coordinates corresponding to the robot-relative positions
    
    Examples
    --------
    # Single point 2 meters forward from robot
    x, y = robot2world(0, 2, rob_x, rob_y, rob_yaw)
    
    # Multiple points at different azimuths
    azimuths = [0, 90, 180, 270]  # forward, right, backward, left
    distances = [1, 1, 1, 1]      # 1 meter each
    x, y = robot2world(azimuths, distances, rob_x, rob_y, rob_yaw)
    """
    # Convert inputs to numpy arrays for consistent handling
    az_deg = np.asarray(az_deg, dtype=float)
    dist = np.asarray(dist, dtype=float)

    # Convert azimuth and robot yaw to radians
    az_rad = np.deg2rad(az_deg)
    rob_yaw_rad = np.deg2rad(rob_yaw_deg)

    # Calculate robot-relative coordinates from azimuth and distance
    # In robot frame: 0° azimuth = forward (positive X in robot frame)
    #                90° azimuth = up/left (positive Y in robot frame)
    # This follows mathematical convention (counter-clockwise from forward)
    # Note: This means 90° is UP/LEFT, not RIGHT, due to standard trigonometry
    x_rel = dist * np.cos(az_rad)  # cos(az) gives X component (forward)
    y_rel = dist * np.sin(az_rad)  # sin(az) gives Y component (up/left)

    # Rotate from robot frame to world frame (rotate by -yaw to undo the world2robot rotation)
    # This is the inverse of the world2robot rotation
    # world2robot uses: x_rel = c*dx + s*dy, y_rel = -s*dx + c*dy (rotation by -yaw)
    # robot2world should use: x_world = c*x_rel - s*y_rel, y_world = s*x_rel + c*y_rel (rotation by +yaw)
    c = np.cos(rob_yaw_rad)
    s = np.sin(rob_yaw_rad)
    x_world = c * x_rel - s * y_rel
    y_world = s * x_rel + c * y_rel

    # Translate to world coordinates
    x_world += rob_x
    y_world += rob_y

    return x_world, y_world


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

def flatten_sonar(sonar_data):
    sonar_data = sonar_data.transpose(0, 2, 1)
    sonar_data = sonar_data.reshape(sonar_data.shape[0], -1)
    return sonar_data

class DataCollection:
    def __init__(self, session_paths, cache_dir=None, force_recompute=False):
        """
        Initialize DataCollection with caching support.
        
        Parameters
        ----------
        session_paths : list
            List of paths to session directories
        cache_dir : str, optional
            Directory for caching processed data. If None, uses './cache' in working directory.
            Set to False to disable caching entirely.
        force_recompute : bool, optional
            If True, clears any existing cache and forces recomputation of all data.
            Useful for development and when cache might be stale.
            
        Notes
        -----
        This class provides efficient access to multi-session data with optional
        disk caching to avoid recomputing expensive operations (profiles, views).
        
        Cache is stored in a visible 'cache' folder by default, making it easy to
        manage and inspect cache contents.
        """
        self.session_paths = session_paths
        
        # Set default cache directory to './cache' if not specified
        if cache_dir is None:
            self.cache_dir = './cache'
            print(f"💾 Using default cache directory: {self.cache_dir}")
        elif cache_dir is False:
            self.cache_dir = None  # Disable caching
            print("💾 Caching disabled")
        else:
            self.cache_dir = cache_dir
            print(f"💾 Using specified cache directory: {self.cache_dir}")
        
        self.processors = []
        self._loaded_profiles = False
        self._loaded_views = False
        self._loaded_sonar = False
        self._loaded_curvatures = False
        self._curvature_distance_cutoff_mm = None
        
        # Clear cache if force_recompute is True and caching is enabled
        if force_recompute and self.cache_dir:
            print(f"🔥 Force recompute enabled - clearing cache directory: {self.cache_dir}")
            self.clear_cache()
        
        # Initialize processors (minimal setup)
        for session_path in session_paths:
            data_reader = DataReader(session_path)
            processor = DataProcessor(data_reader, cache_dir=self.cache_dir, force_recompute=force_recompute)
            self.processors.append(processor)
        
        # Cache metadata
        self._cache_metadata = {
            'session_paths': session_paths,
            'processor_count': len(self.processors),
            'total_samples': sum(p.n for p in self.processors),
            'force_recompute': force_recompute
        }
        
        print(f"📁 DataCollection initialized with {len(self.processors)} sessions")
        print(f"   Total samples: {self._cache_metadata['total_samples']}")
        if cache_dir:
            print(f"   Cache directory: {cache_dir}")
        if force_recompute:
            print(f"   🔥 Force recompute: ENABLED (cache will be ignored)")

    def _get_cache_path(self, data_type, session_index=None):
        """Get cache file path for specific data type."""
        if not self.cache_dir:
            return None
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if session_index is not None:
            session_name = os.path.basename(self.session_paths[session_index])
            return os.path.join(self.cache_dir, f"{session_name}_{data_type}.npy")
        else:
            return os.path.join(self.cache_dir, f"collection_{data_type}.npy")

    def _save_to_cache(self, data, data_type, session_index=None):
        """Save data to cache file."""
        cache_path = self._get_cache_path(data_type, session_index)
        if cache_path:
            np.save(cache_path, data)
            return True
        return False

    def _load_from_cache(self, data_type, session_index=None):
        """Load data from cache file if available."""
        cache_path = self._get_cache_path(data_type, session_index)
        if cache_path and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                print(f"💾 Loaded {data_type} from cache: {cache_path}")
                return data
            except Exception as e:
                print(f"⚠️  Cache load failed for {data_type}: {e}")
        return None

    def load_profiles(self, opening_angle, steps=20, force_recompute=False, fill_nans=True, profile_method='min_bin'):
        """
        Load distance profiles for all sessions with caching.
        
        Parameters
        ----------
        opening_angle : float
            Opening angle in degrees (uses +/- opening_angle/2)
        steps : int
            Number of azimuth steps
        force_recompute : bool
            If True, recompute even if cache exists
        fill_nans : bool
            If True, fill NaNs in profiles. If False, keep NaNs (useful for masks).
        profile_method : str
            Profile extraction method:
            - 'min_bin': minimum distance among wall points within each azimuth bin (legacy behavior).
            - 'ray_center': approximate ray-cast at each bin center using a line-distance tolerance.
            
        Returns
        -------
        tuple
            (profiles, centers) where profiles is (N, steps) and centers is (N, steps)
        """
        # Check if we should force recompute (either globally or for this call)
        global_force_recompute = self._cache_metadata.get('force_recompute', False)
        effective_force_recompute = force_recompute or global_force_recompute
        
        requested_params = {
            'opening_angle': opening_angle,
            'steps': steps,
            'fill_nans': fill_nans,
            'profile_method': profile_method,
        }
        if self._loaded_profiles and not effective_force_recompute:
            if getattr(self, '_profiles_params', None) == requested_params:
                print("📊 Using already loaded profiles")
                return self.get_profiles(), self.get_centers()
            print("📊 Loaded profiles params differ from request; recomputing collection profiles.")
        
        # Compute profiles using processor-level caching
        print(
            f"📊 Loading distance profiles (opening_angle: {opening_angle}°, "
            f"{steps} steps, method={profile_method})..."
        )
        
        all_profiles = []
        all_centers = []
        
        for i, processor in enumerate(self.processors):
            print(f"   Processing session {i+1}/{len(self.processors)}...")
            processor.load_profiles(
                opening_angle=opening_angle,
                steps=steps,
                force_recompute=effective_force_recompute,
                fill_nans=fill_nans,
                profile_method=profile_method,
            )
            
            all_profiles.append(processor.profiles)
            all_centers.append(processor.profile_centers)
        
        # Concatenate all results
        profiles = np.concatenate(all_profiles, axis=0)
        centers = np.concatenate(all_centers, axis=0)
        
        self._loaded_profiles = True
        self._profiles_params = requested_params
        print(f"✅ Loaded profiles for {len(profiles)} total samples")
        print(f"   Profile shape: {profiles.shape}")
        print(f"   Profile centers shape: {centers.shape}")
        
        return profiles, centers

    def load_views(self, radius_mm=1500, opening_angle=90, output_size=(128, 128), force_recompute=False, show_example=True):
        """
        Load conical views for all sessions with caching.
        
        Parameters
        ----------
        radius_mm : float
            Radius of views in millimeters
        opening_angle : float
            Opening angle of views in degrees
        output_size : tuple
            Output size as (width, height)
        force_recompute : bool
            If True, recompute even if cache exists
        show_example : bool
            If True, shows an example view image. Set to False for batch processing.
            
        Returns
        -------
        numpy.ndarray
            Views as uint8 array (N, H, W, 3)
        """
        # Check if we should force recompute (either globally or for this call)
        global_force_recompute = self._cache_metadata.get('force_recompute', False)
        effective_force_recompute = force_recompute or global_force_recompute
        
        if self._loaded_views and not effective_force_recompute:
            print("🎯 Using already loaded views")
            return self.get_views()
        
        # Compute views using processor-level caching
        print(f"🎯 Loading conical views (radius: {radius_mm}mm, opening: {opening_angle}°)...")
        
        all_views = []
        
        for i, processor in enumerate(self.processors):
            print(f"   Processing session {i+1}/{len(self.processors)}...")
            processor.load_views(radius_mm, opening_angle, output_size=output_size, show_example=show_example, force_recompute=effective_force_recompute)
            
            all_views.append(processor.views)
        
        # Concatenate all results
        views = np.concatenate(all_views, axis=0)
        
        self._loaded_views = True
        print(f"✅ Loaded views for {len(views)} total samples")
        print(f"   Views shape: {views.shape}")
        print(f"   Memory usage: {views.nbytes / (1024*1024):.2f} MB")
        
        return views

    def load_sonar(self, flatten=False, force_recompute=False):
        """
        Load sonar data for all sessions.
        
        Parameters
        ----------
        flatten : bool, optional
            If True, flattens the sonar data from (N, samples, 2) to (N, samples*2)
        force_recompute : bool, optional
            If True, recompute even if cache exists
             
        Returns
        -------
        numpy.ndarray
            Sonar data array (N, samples, 2) or (N, samples*2) if flattened
        """
        # Check if we should force recompute (either globally or for this call)
        global_force_recompute = self._cache_metadata.get('force_recompute', False)
        effective_force_recompute = force_recompute or global_force_recompute
        
        if self._loaded_sonar and not effective_force_recompute:
            print("📡 Using already loaded sonar data")
            return self.sonar
        
        # Compute sonar data using processor-level caching
        print(f"📡 Loading sonar data (flatten={flatten})...")
        
        all_sonar = []
        
        for i, processor in enumerate(self.processors):
            print(f"   Processing session {i+1}/{len(self.processors)}...")
            processor.load_sonar(flatten=flatten, force_recompute=effective_force_recompute)
            all_sonar.append(processor.sonar_data)
        
        # Concatenate all results
        sonar_data = np.concatenate(all_sonar, axis=0)
        
        self._loaded_sonar = True
        print(f"✅ Loaded sonar data for {len(sonar_data)} total samples")
        print(f"   Sonar shape: {sonar_data.shape}")
        print(f"   Memory usage: {sonar_data.nbytes / (1024*1024):.2f} MB")
        
        return sonar_data

    def load_curvatures(
        self,
        distance_cutoff_mm,
        force_recompute=False,
        steering_config=None,
        show_progress=True,
        parallel=True,
        num_workers=None,
        backend='thread',
    ):
        """
        Load geometry-derived curvature targets for all sessions.

        Notes
        -----
        Requires profiles to be loaded first via load_profiles(...), so the same
        profile settings are reused.
        """
        global_force_recompute = self._cache_metadata.get('force_recompute', False)
        effective_force_recompute = force_recompute or global_force_recompute
        distance_cutoff_mm = float(distance_cutoff_mm)

        if (
            self._loaded_curvatures
            and not effective_force_recompute
            and self._curvature_distance_cutoff_mm is not None
            and float(self._curvature_distance_cutoff_mm) == distance_cutoff_mm
        ):
            print(f"🌀 Using already loaded curvatures (cutoff={distance_cutoff_mm:.1f} mm)")
            return self.curvatures

        if not self._loaded_profiles:
            raise ValueError("Profiles not loaded. Call load_profiles() before load_curvatures().")

        print(f"🌀 Loading curvature targets (cutoff={distance_cutoff_mm:.1f} mm)...")
        all_curvatures = []
        for i, processor in enumerate(self.processors):
            print(f"   Processing session {i+1}/{len(self.processors)}...")
            curv = processor.load_curvatures(
                distance_cutoff_mm=distance_cutoff_mm,
                force_recompute=effective_force_recompute,
                steering_config=steering_config,
                show_progress=show_progress,
                parallel=parallel,
                num_workers=num_workers,
                backend=backend,
            )
            all_curvatures.append(curv)

        curvatures = np.concatenate(all_curvatures, axis=0).astype(np.float32)
        self._loaded_curvatures = True
        self._curvature_distance_cutoff_mm = distance_cutoff_mm
        print(f"✅ Loaded curvatures for {len(curvatures)} total samples")
        print(f"   Curvatures shape: {curvatures.shape}")
        return curvatures

    def get_field(self, *fields):
        """
        Get concatenated field data from all processors.
        
        Parameters
        ----------
        *fields : str
            Path of keys to traverse (e.g., 'position', 'x' for nested access)
            
        Returns
        -------
        numpy.ndarray
            Concatenated field data
            
        Notes
        -----
        This method gets data from all processors in the collection.
        For filename-specific filtering, use the DataProcessor.get_field() method
        on individual processors or create a DataCollection with only the desired sessions.
        """
        collection = []
        for processor in self.processors:
            if len(fields) == 1 and hasattr(processor, fields[0]):
                # Direct attribute access for single fields
                data = getattr(processor, fields[0])
                collection.append(data)
            else:
                # Try to get from data reader (supports nested fields)
                try:
                    data = processor.get_field(*fields)
                    collection.append(data)
                except:
                    raise ValueError(f"Field '{'.'.join(fields)}' not found in processors")
        
        if not collection:
            raise ValueError(f"No data found for field '{'.'.join(fields)}'")
            
        # Handle different data types appropriately
        if isinstance(collection[0], np.ndarray):
            data = np.concatenate(collection, axis=0)
        else:
            data = np.array(collection)
            
        return np.asarray(data, dtype=np.float32)

    @property
    def profiles(self):
        """
        Get concatenated distance profiles.
        
        Returns
        -------
        numpy.ndarray
        Profiles array (N, steps)
            
        Raises
        ------
        ValueError
            If profiles not loaded
        """
        if not self._loaded_profiles:
            raise ValueError("Profiles not loaded. Call load_profiles() first.")
        
        return np.concatenate([p.profiles for p in self.processors], axis=0)

    @property
    def profile_centers(self):
        """
        Get azimuth centers for profiles.
        
        Returns
        -------
        numpy.ndarray
        Centers array (N, steps)
            
        Raises
        ------
        ValueError
            If profiles not loaded
        """
        if not self._loaded_profiles:
            raise ValueError("Profiles not loaded. Call load_profiles() first.")
        
        return np.concatenate([p.profile_centers for p in self.processors], axis=0)

    @property
    def views(self):
        """
        Get concatenated conical views.
        
        Returns
        -------
        numpy.ndarray
            Views as uint8 array (N, H, W, 3)
            
        Raises
        ------
        ValueError
            If views not loaded
        """
        if not self._loaded_views:
            raise ValueError("Views not loaded. Call load_views() first.")
        
        return np.concatenate([p.views for p in self.processors], axis=0)

    @property
    def sonar(self):
        """
        Get concatenated sonar data.
        
        Returns
        -------
        numpy.ndarray
            Sonar data array (N, samples, 2) or (N, samples*2) if flattened
             
        Raises
        ------
        ValueError
            If sonar not loaded
        """
        if not self._loaded_sonar:
            raise ValueError("Sonar not loaded. Call load_sonar() first.")
         
        return np.concatenate([p.sonar_data for p in self.processors], axis=0)

    @property
    def curvatures(self):
        """Get concatenated curvature targets for all sessions."""
        if not self._loaded_curvatures:
            raise ValueError("Curvatures not loaded. Call load_curvatures() first.")
        return np.concatenate([p.curvatures for p in self.processors], axis=0)

    @property
    def rob_x(self):
        """
        Get concatenated robot X positions from all sessions.
        
        Returns
        -------
        numpy.ndarray
            Array of robot X positions in millimeters
        """
        return np.concatenate([p.rob_x for p in self.processors], axis=0)

    @property
    def rob_y(self):
        """
        Get concatenated robot Y positions from all sessions.
        
        Returns
        -------
        numpy.ndarray
            Array of robot Y positions in millimeters
        """
        return np.concatenate([p.rob_y for p in self.processors], axis=0)

    @property
    def rob_yaw_deg(self):
        """
        Get concatenated robot yaw orientations from all sessions.
        
        Returns
        -------
        numpy.ndarray
            Array of robot yaw orientations in degrees
        """
        return np.concatenate([p.rob_yaw_deg for p in self.processors], axis=0)
    
    @property
    def quadrants(self):
        """
        Get concatenated spatial quadrants from all sessions.
        
        Returns
        -------
        numpy.ndarray
            Array of quadrant indices (0-3) for each data point
            
        Raises
        ------
        ValueError
            If quadrants not available in all processors
        """
        # Check if all processors have quadrants
        for processor in self.processors:
            if not hasattr(processor, 'quadrants') or processor.quadrants is None:
                raise ValueError("Quadrants not available in all processors")
        
        return np.concatenate([p.quadrants for p in self.processors], axis=0)

    def save_cache(self):
        """
        Save all loaded data to cache.
        
        Notes
        -----
        Caching is now handled at the processor level, so this method
        just ensures all processors have saved their cache.
        """
        if not self.cache_dir:
            print("⚠️  No cache directory specified")
            return
            
        print(f"💾 Saving cache to {self.cache_dir}...")
        
        # Caching is now handled at processor level, no need to save here
        # Processors automatically save their cache when data is loaded
        
        # Save metadata
        meta_path = os.path.join(self.cache_dir, 'collection_meta.json')
        with open(meta_path, 'w') as f:
            json.dump(self._cache_metadata, f, indent=2)
        print("   ✅ Saved metadata")
        
        print("🎉 Cache saved successfully!")

    def clear_cache(self):
        """
        Clear all cache files.
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
            
        print(f"🧹 Clearing cache directory: {self.cache_dir}")
        for file in os.listdir(self.cache_dir):
            if file.startswith('collection_') or any(sess in file for sess in 
                [os.path.basename(path) for path in self.session_paths]):
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                    print(f"   🗑️  Removed: {file}")
                except:
                    pass
        print("✅ Cache cleared")



class DataProcessor:
    def __init__(self, data_reader, cache_dir=None, force_recompute=False):
        """
        Initialize DataProcessor with caching support.
        
        Parameters
        ----------
        data_reader : str or DataReader
            Path to data directory or DataReader instance
        cache_dir : str, optional
            Directory for caching processed data. If None, uses './cache' in working directory.
            Set to False to disable caching entirely.
        force_recompute : bool, optional
            If True, clears any existing cache and forces recomputation of all data.
            Useful for development and when cache might be stale.
            
        Notes
        -----
        For additional functionality, explicitly call:
        - load_arena() for arena metadata (needed for profiles/views)
        - load_profiles(opening_angle, steps) for distance profiles
        - load_views(radius_mm, opening_angle, output_size) for conical views
        """
        if type(data_reader) == str: data_reader = DataReader(data_reader)
        self.session = data_reader.base_folder
        self.data_reader = data_reader
        self.env_dir = get_env_dir(data_reader)
        self.wall_mask, self.path_mask, self.meta = None, None, None
        self.wall_x, self.wall_y = None, None
        self.path_x, self.path_y = None, None
        self.arena_loaded = False
        self.arena_metadata_loaded = False
        
        # Cache configuration
        if cache_dir is None:
            self.cache_dir = './cache'
        elif cache_dir is False:
            self.cache_dir = None  # Disable caching
        else:
            self.cache_dir = cache_dir
        
        self.force_recompute = force_recompute
        
        # Profile parameters (set when load_profiles is called)
        self.profiles_loaded = False
        self.profiles = None
        self.profile_centers = None
        self.profile_opening_angle = None
        self.profile_steps = None
        self.profile_fill_nans = None
        
        # View parameters (set when load_views is called)
        self.views_loaded = False
        self.views = None
        
        # Sonar parameters (set when load_sonar is called)
        self.sonar_loaded = False
        self.sonar_data = None

        # Curvature targets (set when load_curvatures is called)
        self.curvatures_loaded = False
        self.curvatures = None
        self.curvature_distance_cutoff_mm = None
        
        self.positions = read_robot_trajectory(data_reader)
        self.filenames = self.data_reader.get_all_filenames()
        self.n = len(self.filenames)
        self.rob_x = self.positions.rob_x
        self.rob_y = self.positions.rob_y
        self.rob_yaw_deg = self.positions.rob_yaw_deg
        self.missing = self.positions.missing
        
        # Calculate spatial quadrants based on mean x and y positions
        self._calculate_spatial_quadrants()
        
        # Clear cache if force_recompute is True and caching is enabled
        if self.force_recompute and self.cache_dir:
            self.clear_cache()
        
        print(f"💾 DataProcessor initialized for session: {os.path.basename(self.session)}")
        if self.cache_dir:
            print(f"   Cache directory: {self.cache_dir}")
        else:
            print("   Caching disabled")
        if self.force_recompute:
            print("   🔥 Force recompute: ENABLED")

    def _get_cache_path(self, data_type):
        """Get cache file path for specific data type."""
        if not self.cache_dir:
            return None
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        session_name = os.path.basename(self.session)
        return os.path.join(self.cache_dir, f"{session_name}_{data_type}.npy")

    def _save_to_cache(self, data, data_type, metadata=None):
        """Save data to cache file with optional metadata."""
        cache_path = self._get_cache_path(data_type)
        if cache_path:
            if metadata is not None:
                # Save data and metadata as a dictionary using Python pickle
                import pickle
                cache_data = {'data': data, 'metadata': metadata}
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            else:
                np.save(cache_path, data)
            return True
        return False

    def _load_from_cache(self, data_type):
        """Load data from cache file if available."""
        cache_path = self._get_cache_path(data_type)
        if cache_path and os.path.exists(cache_path):
            try:
                # Try Python pickle first (new format with metadata)
                try:
                    import pickle
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    if isinstance(cache_data, dict) and 'data' in cache_data:
                        # New format with metadata
                        data = cache_data['data']
                        metadata = cache_data.get('metadata', {})
                        print(f"💾 Loaded {data_type} from cache: {cache_path}")
                        return data, metadata
                except:
                    pass
                
                # Fall back to numpy load (old format without metadata)
                cache_data = np.load(cache_path, allow_pickle=True)
                
                # Handle old format (raw array)
                print(f"💾 Loaded {data_type} from cache: {cache_path}")
                return cache_data, {}
            except Exception as e:
                print(f"⚠️  Cache load failed for {data_type}: {e}")
        return None, None

    def clear_cache(self):
        """Clear all cache files for this session."""
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return
        
        session_name = os.path.basename(self.session)
        print(f"🧹 Clearing cache for session: {session_name}")
        
        for file in os.listdir(self.cache_dir):
            if file.startswith(f"{session_name}_"):
                try:
                    os.remove(os.path.join(self.cache_dir, file))
                    print(f"   🗑️  Removed: {file}")
                except:
                    pass
        print("✅ Cache cleared")

    def print_data_overview(self):
        print('DATA IN DATA READER:')
        self.data_reader.print_data_overview()

    def get_motion(self):
        distances = self.data_reader.get_field('motion', 'distance')
        rotations = self.data_reader.get_field('motion', 'rotation')
        distances = np.asarray(distances)
        rotations = np.asarray(rotations)
        return distances, rotations

    def get_field(self, *fields, filenames=None):
        """
        Get field data from data reader.
        
        Parameters
        ----------
        *fields : str
            Path of keys to traverse (e.g., 'position', 'x')
        filenames : list of str, optional
            Specific filenames to load. If None, all files are loaded.
            
        Returns
        -------
        numpy.ndarray
            Field data
        """
        if filenames is not None:
            values = self.data_reader.get_field(*fields, filenames=filenames)
        else:
            values = self.data_reader.get_field(*fields)
        values = np.asarray(values)
        return values

    def copy_annotated(self, original_path):
        dest_path = self.env_dir / "arena_mask_annotated.png"
        shutil.copy(original_path, dest_path)
        self.load_arena()

    def _load_meta_only(self):
        """
        Load only meta.json file (for coordinate conversion).
        
        This method loads just the meta data without requiring
        arena_annotated.png, which is sufficient for view extraction.
        
        Required for: view extraction coordinate conversion
        """
        try:
            env_dir = get_env_dir(self.data_reader)
            meta_path = env_dir / "meta.json"
            self.meta = json.load(open(meta_path))
        except FileNotFoundError:
            print('Could not find meta.json file')
            
    def load_arena_metadata(self):
        """
        Load arena metadata (masks and coordinates) from annotated image.
        
        This method loads the arena annotation data including:
        - Wall and path masks from arena_annotated.png
        - Meta data from meta.json
        - Derived wall and path coordinates
        
        Required for: distance profile computation
        
        Notes
        -----
        This does NOT load the arena.png image - that's handled separately
        by load_arena_image() for view extraction.
        """
        try:
            self.wall_mask, self.path_mask, self.meta = load_arena_masks(self.data_reader)
            self.wall_x, self.wall_y = mask2coordinates(self.wall_mask, self.meta)
            self.path_x, self.path_y = mask2coordinates(self.path_mask, self.meta)
            self.arena_metadata_loaded = True
            print("✅ Arena metadata loaded successfully")
        except FileNotFoundError:
            print('❌ Could not find the arena annotation files')
            print('   wall_x and wall_y will remain None')
            self.arena_metadata_loaded = False
            
    def load_arena_image(self):
        """
        Load the actual arena overhead image.
        
        This method loads the arena.png image for view extraction.
        
        Required for: conical view extraction
        
        Returns
        -------
        numpy.ndarray
            Arena image as RGB numpy array (H, W, 3)
        
        Notes
        -----
        This does NOT load arena metadata - that's handled separately
        by load_arena_metadata() for profile computation.
        """
        arena_image_path = self.env_dir / "arena.png"
        if not arena_image_path.exists():
            raise FileNotFoundError(f"Arena image not found at {arena_image_path}")
            
        img_bgr = cv2.imread(str(arena_image_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb

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

        for i in range(0, self.n, 5): plt.text(self.rob_x[i], self.rob_y[i], str(i), color='green', fontsize=12)
        if show: plt.show()
        return plt.gcf()

    def get_arena_extent(self):
        min_x = self.meta["arena_bounds_mm"]["min_x"]
        max_x = self.meta["arena_bounds_mm"]["max_x"]
        min_y = self.meta["arena_bounds_mm"]["min_y"]
        max_y = self.meta["arena_bounds_mm"]["max_y"]
        extent = (min_x, max_x, min_y, max_y)
        return extent

    def world_to_pixel(self, x_world, y_world):
        """
        Convert world coordinates (mm) to pixel coordinates in arena image.
        
        Parameters
        ----------
        x_world, y_world : float
            World coordinates in millimeters
            
        Returns
        -------
        x_pixel, y_pixel : int
            Pixel coordinates in arena image
        """
        min_x = self.meta["arena_bounds_mm"]["min_x"]
        max_y = self.meta["arena_bounds_mm"]["max_y"]
        mm_per_px = float(self.meta["map_mm_per_px"])
        
        x_pixel = int(round((x_world - min_x) / mm_per_px))
        y_pixel = int(round((max_y - y_world) / mm_per_px))
        
        return x_pixel, y_pixel

    def pixel_to_world(self, x_pixel, y_pixel):
        """
        Convert pixel coordinates to world coordinates (mm).
        
        Parameters
        ----------
        x_pixel, y_pixel : int
            Pixel coordinates in arena image
            
        Returns
        -------
        x_world, y_world : float
            World coordinates in millimeters
        """
        min_x = self.meta["arena_bounds_mm"]["min_x"]
        max_y = self.meta["arena_bounds_mm"]["max_y"]
        mm_per_px = float(self.meta["map_mm_per_px"])
        
        x_world = min_x + x_pixel * mm_per_px + 0.5 * mm_per_px
        y_world = max_y - y_pixel * mm_per_px + 0.5 * mm_per_px
        
        return x_world, y_world

    def load_profiles(self, opening_angle, steps=20, force_recompute=False, fill_nans=True, profile_method='min_bin'):
        """
        Load distance profiles for all robot positions.
        
        Parameters
        ----------
        opening_angle : float
            Opening angle in degrees (uses +/- opening_angle/2)
        steps : int
            Number of azimuth steps in the profile
        force_recompute : bool
            If True, recompute even if cache exists
        fill_nans : bool
            If True, fill NaNs in profiles. If False, keep NaNs (useful for masks).
        profile_method : str
            Profile extraction method:
            - 'min_bin': minimum distance among wall points within each azimuth bin (legacy behavior).
            - 'ray_center': approximate ray-cast at each bin center using line-distance tolerance.
            
        Notes
        -----
        Automatically loads arena metadata if needed.
        Results are cached for efficient reuse.
        """
        az_min = -0.5 * opening_angle
        az_max = 0.5 * opening_angle

        # Check if we should force recompute (either globally or for this call)
        effective_force_recompute = force_recompute or self.force_recompute
        
        if self.profiles_loaded and not effective_force_recompute:
            already_match = (
                getattr(self, 'profile_opening_angle', None) == opening_angle
                and getattr(self, 'profile_steps', None) == steps
                and getattr(self, 'profile_fill_nans', None) == fill_nans
                and getattr(self, 'profile_method', 'min_bin') == profile_method
            )
            if already_match:
                print("📊 Using already loaded profiles")
                return
            print("📊 Loaded profiles params differ from request; recomputing session profiles.")
            
        # Try to load from cache first (only if not forcing recompute)
        if not effective_force_recompute:
            cached_profiles, profiles_metadata = self._load_from_cache('profiles')
            cached_centers, centers_metadata = self._load_from_cache('centers')
            
            # Check if cached data exists and parameters match
            if (cached_profiles is not None and cached_centers is not None and
                profiles_metadata is not None and centers_metadata is not None):
                
                # Extract parameters from metadata
                cached_params = {
                    'opening_angle': profiles_metadata.get('opening_angle'),
                    'steps': profiles_metadata.get('steps'),
                    'fill_nans': profiles_metadata.get('fill_nans', True),
                    'profile_method': profiles_metadata.get('profile_method', 'min_bin'),
                }
                
                current_params = {
                    'opening_angle': opening_angle,
                    'steps': steps,
                    'fill_nans': fill_nans,
                    'profile_method': profile_method,
                }
                
                # Check if parameters match - handle numpy array comparisons safely
                def params_equal(a, b):
                    """Safe comparison that handles numpy arrays and other types"""
                    try:
                        # Try direct comparison first
                        return a == b
                    except (ValueError, TypeError):
                        # If direct comparison fails (e.g., numpy array), try alternative methods
                        if hasattr(a, '__array__') and hasattr(b, '__array__'):
                            # Both are numpy arrays, use numpy's array_equal
                            import numpy as np
                            return np.array_equal(a, b)
                        elif hasattr(a, '__iter__') and hasattr(b, '__iter__'):
                            # Both are iterable, convert to tuples and compare
                            return tuple(a) == tuple(b)
                        else:
                            # Fallback to string comparison if all else fails
                            return str(a) == str(b)
                
                params_match = all(
                    params_equal(cached_params.get(k), v) 
                    for k, v in current_params.items()
                )
                
                if params_match:
                    self.profiles = cached_profiles
                    self.profile_centers = cached_centers
                    self.profiles_loaded = True
                    self.profile_opening_angle = opening_angle
                    self.profile_steps = steps
                    self.profile_fill_nans = fill_nans
                    self.profile_method = profile_method
                    print(f"✅ Loaded {len(self.profiles)} distance profiles from cache")
                    print(f"   Profile shape: {self.profiles.shape}")
                    print(f"   Centers shape: {self.profile_centers.shape}")
                    print(
                        f"   Parameters matched: opening_angle={opening_angle}, "
                        f"steps={steps}, method={profile_method}"
                    )
                    
                    # Ensure arena metadata is loaded even when using cached profiles
                    if not hasattr(self, 'arena_metadata_loaded') or not self.arena_metadata_loaded:
                        self.load_arena_metadata()
                    return
                else:
                    print(f"⚠️  Cache parameters don't match - recomputing...")
                    print(f"   Cached params: {cached_params}")
                    print(f"   Requested params: {current_params}")
        
        # Automatically load arena metadata if needed (for wall coordinates)
        if not hasattr(self, 'arena_metadata_loaded') or not self.arena_metadata_loaded:
            self.load_arena_metadata()
            
        # Check if arena metadata loaded successfully
        if not self.arena_metadata_loaded:
            # Try to load arena metadata one more time in case it was missed
            self.load_arena_metadata()
            
            if not self.arena_metadata_loaded:
                raise RuntimeError("Cannot compute distance profiles - arena metadata failed to load. "
                                 "Make sure arena_annotated.png and meta.json exist in the environment directory.")
        
        print(
            f"📊 Computing distance profiles (opening_angle: {opening_angle}°, "
            f"{steps} steps, method={profile_method})..."
        )
        
        # Compute profiles for all positions using parallel processing
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def compute_profile_wrapper(index):
            """Wrapper function for parallel profile computation"""
            centers_i, profile_i = self.get_profile_at(
                index,
                az_min,
                az_max,
                steps,
                fill_nans=fill_nans,
                profile_method=profile_method,
            )
            return index, centers_i, profile_i
        
        # Use ThreadPoolExecutor for parallel processing
        num_workers = min(multiprocessing.cpu_count(), self.n)
        
        print(f"   🚀 Using {num_workers} workers for parallel profile computation")
        
        # Pre-allocate lists
        profiles = [None] * self.n
        centers = [None] * self.n
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(compute_profile_wrapper, index): index 
                              for index in range(self.n)}
            
            # Process completed tasks as they come in
            for future in tqdm(as_completed(future_to_index), total=self.n, desc="Computing profiles"):
                index, centers_i, profile_i = future.result()
                profiles[index] = profile_i
                centers[index] = centers_i
        
        self.profiles = np.asarray(profiles, dtype=np.float32)
        self.profile_centers = np.asarray(centers, dtype=np.float32)
        self.profiles_loaded = True
        self.profile_opening_angle = opening_angle
        self.profile_steps = steps
        self.profile_fill_nans = fill_nans
        self.profile_method = profile_method
        
        # Save to cache with metadata
        if self.cache_dir:
            from datetime import datetime
            profiles_metadata = {
                'opening_angle': opening_angle,
                'steps': steps,
                'fill_nans': fill_nans,
                'profile_method': profile_method,
                'timestamp': datetime.now().isoformat()
            }
            centers_metadata = {
                'opening_angle': opening_angle,
                'steps': steps,
                'fill_nans': fill_nans,
                'profile_method': profile_method,
                'timestamp': datetime.now().isoformat()
            }
            self._save_to_cache(self.profiles, 'profiles', metadata=profiles_metadata)
            self._save_to_cache(self.profile_centers, 'centers', metadata=centers_metadata)
        
        print(f"✅ Loaded {len(self.profiles)} distance profiles")
        print(f"   Profile shape: {self.profiles.shape}")
        print(f"   Centers shape: {self.profile_centers.shape}")

    def load_curvatures(
        self,
        distance_cutoff_mm,
        force_recompute=False,
        steering_config=None,
        show_progress=True,
        parallel=True,
        num_workers=None,
        backend='thread',
    ):
        """
        Load per-sample curvature targets derived from currently loaded profiles.

        Parameters
        ----------
        distance_cutoff_mm : float
            Distances above this threshold are ignored (set to NaN) before
            curvature computation.
        force_recompute : bool
            If True, recompute even if cached.
        steering_config : SteeringConfig, optional
            Planner settings for profile2curvature; defaults to SteeringConfig().
        show_progress : bool
            If True, show tqdm progress.
        parallel : bool
            If True, compute targets using a worker pool.
        num_workers : int or None
            Number of workers for parallel mode. Defaults to min(cpu_count, n).
        backend : str
            Parallel backend: 'thread' or 'process'.
        """
        effective_force_recompute = force_recompute or self.force_recompute
        distance_cutoff_mm = float(distance_cutoff_mm)

        if (
            self.curvatures_loaded
            and not effective_force_recompute
            and self.curvature_distance_cutoff_mm is not None
            and float(self.curvature_distance_cutoff_mm) == distance_cutoff_mm
        ):
            print(f"🌀 Using already loaded curvature targets (cutoff={distance_cutoff_mm:.1f} mm)")
            return self.curvatures

        if not self.profiles_loaded or self.profiles is None or self.profile_centers is None:
            raise ValueError("Profiles not loaded. Call load_profiles(...) before load_curvatures(...).")

        cfg = steering_config
        if cfg is None:
            from Library.SteeringConfigClass import SteeringConfig
            cfg = SteeringConfig()

        # Try cache.
        if not effective_force_recompute:
            cached_curv, cached_meta = self._load_from_cache('curvatures')
            if cached_curv is not None and cached_meta is not None:
                cached_cutoff = cached_meta.get('distance_cutoff_mm', None)
                cached_opening = cached_meta.get('profile_opening_angle', None)
                cached_steps = cached_meta.get('profile_steps', None)
                if (
                    cached_cutoff is not None and float(cached_cutoff) == distance_cutoff_mm
                    and cached_opening == self.profile_opening_angle
                    and cached_steps == self.profile_steps
                    and len(cached_curv) == self.n
                ):
                    self.curvatures = np.asarray(cached_curv, dtype=np.float32)
                    self.curvatures_loaded = True
                    self.curvature_distance_cutoff_mm = distance_cutoff_mm
                    print(f"✅ Loaded {len(self.curvatures)} curvature targets from cache")
                    return self.curvatures

        print(f"🌀 Computing curvature targets (cutoff={distance_cutoff_mm:.1f} mm)...")
        curvatures = np.zeros(self.n, dtype=np.float32)
        n_workers = int(num_workers) if num_workers is not None else min(os.cpu_count() or 1, self.n)
        use_parallel = bool(parallel) and n_workers > 1

        if use_parallel:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
            executor_cls = ProcessPoolExecutor if str(backend).lower() == 'process' else ThreadPoolExecutor
            tasks = [
                (i, self.profile_centers[i], self.profiles[i], distance_cutoff_mm, cfg)
                for i in range(self.n)
            ]
            with executor_cls(max_workers=n_workers) as executor:
                iterator = executor.map(_compute_curvature_single, tasks)
                if show_progress:
                    iterator = tqdm(iterator, total=self.n, desc='Computing curvatures')
                for i, kappa in iterator:
                    curvatures[int(i)] = float(kappa)
        else:
            iterator = range(self.n)
            if show_progress:
                iterator = tqdm(iterator, total=self.n, desc='Computing curvatures')
            for i in iterator:
                _, kappa = _compute_curvature_single(
                    (i, self.profile_centers[i], self.profiles[i], distance_cutoff_mm, cfg)
                )
                curvatures[i] = float(kappa)

        self.curvatures = curvatures
        self.curvatures_loaded = True
        self.curvature_distance_cutoff_mm = distance_cutoff_mm

        if self.cache_dir:
            from datetime import datetime
            metadata = {
                'distance_cutoff_mm': distance_cutoff_mm,
                'profile_opening_angle': self.profile_opening_angle,
                'profile_steps': self.profile_steps,
                'timestamp': datetime.now().isoformat(),
            }
            self._save_to_cache(self.curvatures, 'curvatures', metadata=metadata)

        print(f"✅ Loaded {len(self.curvatures)} curvature targets")
        return self.curvatures

    def load_views(self, radius_mm=1500, opening_angle=90, output_size=(128, 128), plot_indices=None, indices=None, save_examples=False, force_recompute=False, show_example=True):
        """
        Load views for all robot positions.
        
        Parameters
        ----------
        radius_mm : float
            Radius of views in millimeters
        opening_angle : float
            Opening angle of views in degrees
        output_size : tuple
            Output size of views as (width, height)
        plot_indices : list, optional
            If provided, shows visualization plots for these specific indices.
            Replaces the old show_example parameter.
        indices : list, optional
            If provided, only load views for these specific indices. Useful for debugging.
        save_examples : bool or str, optional
            If True, saves example plots for all views.
            If str, saves example plots and uses it as directory name.
            Useful for debugging orientation issues.
        force_recompute : bool
            If True, recompute even if cache exists
            
        Notes
        -----
        Automatically loads arena metadata if needed.
        Results are cached for efficient reuse.
        """
        # Check if we should force recompute (either globally or for this call)
        effective_force_recompute = force_recompute or self.force_recompute
        
        if self.views_loaded and not effective_force_recompute:
            print("🎯 Using already loaded views")
            return self.views
            
        # Try to load from cache first (only if not forcing recompute)
        if not effective_force_recompute:
            cached_views, views_metadata = self._load_from_cache('views')
            
            if cached_views is not None and views_metadata is not None:
                # Extract parameters from metadata
                cached_params = {
                    'radius_mm': views_metadata.get('radius_mm'),
                    'opening_angle': views_metadata.get('opening_angle'),
                    'output_size': tuple(views_metadata.get('output_size', []))
                }
                
                current_params = {
                    'radius_mm': radius_mm,
                    'opening_angle': opening_angle,
                    'output_size': output_size
                }
                
                # Check if parameters match
                params_match = all(
                    cached_params.get(k) == v 
                    for k, v in current_params.items()
                )
                
                if params_match:
                    self.views = cached_views
                    self.views_loaded = True
                    self.view_radius_mm = radius_mm
                    self.view_opening_angle = opening_angle
                    self.view_output_size = output_size
                    print(f"✅ Loaded {len(self.views)} views from cache")
                    print(f"   Views shape: {self.views.shape}")
                    print(f"   Memory usage: {self.views.nbytes / 1024 / 1024:.2f} MB")
                    print(f"   Parameters matched: radius={radius_mm}mm, opening={opening_angle}°, size={output_size}")
                    return self.views
                else:
                    print(f"⚠️  Cache parameters don't match - recomputing...")
                    print(f"   Cached params: {cached_params}")
                    print(f"   Requested params: {current_params}")
        
        # Store parameters
        self.view_radius_mm = radius_mm
        self.view_opening_angle = opening_angle
        self.view_output_size = output_size
        
        print(f"🎯 Computing views (radius: {radius_mm}mm, opening: {opening_angle}°)...")
        
        # Load meta data if needed (for coordinate conversion)
        if not hasattr(self, 'meta') or self.meta is None:
            self._load_meta_only()
        
        # Load arena image if not already loaded
        if not hasattr(self, 'arena_image_cache'):
            self.arena_image_cache = self.load_arena_image()
            self.arena_image_shape = self.arena_image_cache.shape
        
        # Determine which indices to process
        if indices is None:
            indices_to_process = range(self.n)
        else:
            indices_to_process = indices
            print(f"   Processing only indices: {indices}")
        
        # Set up saving for plot_indices
        save_plots = False
        plots_dir = None
        if plot_indices is not None:
            import os
            from datetime import datetime
            
            # Create directory for saved plots
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_name = os.path.basename(self.session)
            plots_dir = os.path.join('Plots', f'conical_views_{session_name}_{timestamp}')
            os.makedirs(plots_dir, exist_ok=True)
            print(f"   💾 Saving visualization plots to: {plots_dir}")
            save_plots = True
        
        # Set up saving if requested (separate from plot_indices)
        if save_examples:
            import os
            from datetime import datetime
            
            # Create directory for saved examples
            if isinstance(save_examples, str):
                examples_dir = save_examples
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                examples_dir = f'conical_view_examples_{timestamp}'
            
            os.makedirs(examples_dir, exist_ok=True)
            print(f"   💾 Saving example plots to: {examples_dir}")
        
        # Extract views for specified positions using parallel processing
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def extract_view_wrapper(index):
            """Wrapper function for parallel view extraction"""
            rob_x_i = self.rob_x[index]
            rob_y_i = self.rob_y[index]
            rob_yaw_i = self.rob_yaw_deg[index]
            
            # Determine if we should visualize and where to save
            should_visualize = (plot_indices is not None and index in plot_indices)
            save_path = None
            if save_plots and should_visualize:
                plot_filename = os.path.join(plots_dir, f'conical_view_{index:04d}.png')
                save_path = plot_filename
            
            view = self.extract_conical_view(
                rob_x_i, rob_y_i, rob_yaw_i,
                radius_mm=self.view_radius_mm,
                opening_angle_deg=self.view_opening_angle,
                output_size=self.view_output_size,
                visualize=should_visualize,
                save_path=save_path
            )
            return index, view
        
        # Use ThreadPoolExecutor for parallel processing
        # Threads are more efficient than processes for this I/O-bound task
        num_workers = min(multiprocessing.cpu_count(), len(indices_to_process))
        
        print(f"   🚀 Using {num_workers} workers for parallel view extraction")
        
        views = [None] * len(indices_to_process)  # Pre-allocate list
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(extract_view_wrapper, index): index 
                              for index in indices_to_process}
            
            # Process completed tasks as they come in
            for future in tqdm(as_completed(future_to_index), total=len(indices_to_process), 
                             desc="Extracting views"):
                index, view = future.result()
                views[index] = view
        
        self.views = np.asarray(views, dtype=np.uint8)
        self.views_loaded = True
        
        # Save to cache with metadata
        if self.cache_dir:
            from datetime import datetime
            views_metadata = {
                'radius_mm': radius_mm,
                'opening_angle': opening_angle,
                'output_size': list(output_size),
                'timestamp': datetime.now().isoformat()
            }
            self._save_to_cache(self.views, 'views', metadata=views_metadata)
        
        print(f"✅ Loaded {len(self.views)} views")
        print(f"   Views shape: {self.views.shape}")
        print(f"   Memory usage: {self.views.nbytes / 1024 / 1024:.2f} MB")
        
        return self.views

    def load_sonar(self, flatten=False, force_recompute=False):
        """
        Load sonar data for all robot positions.
        
        Parameters
        ----------
        flatten : bool, optional
            If True, flattens the sonar data from (N, samples, 2) to (N, samples*2)
            for compatibility with some machine learning models.
        force_recompute : bool
            If True, recompute even if cache exists
           
             
        Notes
        -----
        Automatically extracts left and right channels based on configuration.
        Results are cached for efficient reuse.
        """
        # Check if we should force recompute (either globally or for this call)
        effective_force_recompute = force_recompute or self.force_recompute
        
        if self.sonar_loaded and not effective_force_recompute:
            print("📡 Using already loaded sonar data")
            return
            
        # Try to load from cache first (only if not forcing recompute)
        if not effective_force_recompute:
            cached_sonar, sonar_metadata = self._load_from_cache('sonar')
            
            if cached_sonar is not None and sonar_metadata is not None:
                # Check if flatten parameter matches
                cached_flatten = sonar_metadata.get('flatten', False)
                if cached_flatten == flatten:
                    self.sonar_data = cached_sonar
                    self.sonar_loaded = True
                    print("📡 Using cached sonar data")
                    print(f"   📏 Sonar data shape: {self.sonar_data.shape}")
                    print(f"   Memory usage: {self.sonar_data.nbytes / 1024 / 1024:.2f} MB")
                    print(f"   Parameters matched: flatten={flatten}")
                    return
                else:
                    print(f"⚠️  Cache flatten parameter doesn't match - recomputing...")
                    print(f"   Cached flatten: {cached_flatten}")
                    print(f"   Requested flatten: {flatten}")
            
        print(f"📡 Loading sonar data...")
        
        # Load sonar data for all positions
        sonar_data = self.get_field('sonar_package', 'sonar_data')
        sonar_data = np.array(sonar_data, dtype=np.float32)
        
        # sonar_data is already stored as [emitter, left, right] in Client.read_buffers
        # Keep only the left/right channels by fixed indices.
        sonar_data = sonar_data[:, :, [1, 2]]
        
        if flatten:
            sonar_data = flatten_sonar(sonar_data)
            print(f"   📏 Flattened sonar data to shape: {sonar_data.shape}")
        else:
            print(f"   📏 Sonar data shape: {sonar_data.shape}")
        
        self.sonar_data = sonar_data
        self.sonar_loaded = True
        
        # Save to cache with metadata
        if self.cache_dir:
            from datetime import datetime
            sonar_metadata = {
                'flatten': flatten,
                'timestamp': datetime.now().isoformat()
            }
            self._save_to_cache(self.sonar_data, 'sonar', metadata=sonar_metadata)
        
        print(f"✅ Loaded sonar data for {len(sonar_data)} positions")
        print(f"   Memory usage: {sonar_data.nbytes / 1024 / 1024:.2f} MB")

    def create_conical_mask(self, center_x, center_y, radius_px, opening_angle_deg, orientation_deg=0):
        """
        Create a binary mask for a conical (pie-shaped) region.
        
        Parameters
        ----------
        center_x, center_y : int
            Center of the cone in pixel coordinates
        radius_px : int
            Radius of the cone in pixels
        opening_angle_deg : float
            Opening angle of the cone in degrees
        orientation_deg : float
            Orientation of the cone (0 = right/east, 90 = up/north)
            
        Returns
        -------
        numpy.ndarray
            Binary mask (True inside cone, False outside)
        """
        # Create empty mask
        height, width = self.arena_image_shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        
        # Convert to radians
        opening_angle_rad = np.deg2rad(opening_angle_deg)
        orientation_rad = np.deg2rad(orientation_deg)
        
        # Create grid of coordinates relative to center
        y, x = np.ogrid[:height, :width]
        x_rel = x - center_x
        y_rel = y - center_y
        
        # Calculate distance from center
        distance = np.sqrt(x_rel**2 + y_rel**2)
        
        # Calculate angles relative to center (0 = right/east, increasing counter-clockwise)
        # Note: In image coordinates, y increases DOWNWARD, so we negate y_rel to get
        # the correct angle that matches the mathematical coordinate system used for robot orientation
        angles = np.arctan2(-y_rel, x_rel)  # Back to original -y_rel
        
        # Convert angles from [-π, π] to [0, 2π] range to match robot yaw convention
        angles = np.mod(angles, 2 * np.pi)
        
        # Define cone boundaries
        half_angle = opening_angle_rad / 2
        lower_bound = orientation_rad - half_angle
        upper_bound = orientation_rad + half_angle
        
        # Handle angle wrapping using circular distance
        # Calculate the angular distance from the center orientation
        angle_diff = np.abs(angles - orientation_rad)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)  # Circular distance
        
        in_cone = angle_diff <= half_angle
        
        # Combine with distance constraint
        in_cone = in_cone & (distance <= radius_px)
        
        return in_cone

    def extract_conical_view(self, rob_x, rob_y, rob_yaw_deg, 
                           radius_mm=1500, opening_angle_deg=90,
                           output_size=(128, 128), visualize=False, save_path=None):
        """
        Extract a conical (pie-shaped) view of the arena centered on robot position.
        
        Parameters
        ----------
        rob_x, rob_y : float
            Robot position in world coordinates (mm)
        rob_yaw_deg : float  
            Robot orientation in degrees (0 = right/east, 90 = up/north)
        radius_mm : float
            Radius of the conical view in millimeters
        opening_angle_deg : float
            Opening angle of the cone in degrees
        output_size : tuple
            Target output size (width, height) for the conical view
        visualize : bool
            Whether to show visualization of the extraction process
        save_path : str, optional
            If provided, saves the visualization plot to this path instead of showing it
            
        Returns
        -------
        numpy.ndarray
            Conical view as RGB image (output_size[1], output_size[0], 3)
        """
        # Load arena image if not already loaded
        if not hasattr(self, 'arena_image_cache'):
            self.arena_image_cache = self.load_arena_image()
            self.arena_image_shape = self.arena_image_cache.shape
        
        arena_img = self.arena_image_cache
        
        # Convert robot position to pixel coordinates
        center_x, center_y = self.world_to_pixel(rob_x, rob_y)
        
        # Convert radius from mm to pixels
        mm_per_px = float(self.meta["map_mm_per_px"])
        radius_px = int(round(radius_mm / mm_per_px))
        
        # Create conical mask
        # The conical view should show what's FORWARD from the robot's perspective
        # Use the SAME orientation as the trajectory plot (raw yaw angle)
        # The conical view should point in the same direction as the robot's orientation arrow
        # in the trajectory plot, which uses the raw yaw angle without correction
        conical_mask = self.create_conical_mask(
            center_x, center_y, radius_px, opening_angle_deg, rob_yaw_deg
        )
        
        # Extract the conical region
        conical_region = np.zeros_like(arena_img)
        conical_region[conical_mask] = arena_img[conical_mask]
        
        # NEW APPROACH: Extract all pixels in the conical mask (including black ones at boundaries)
        # This ensures we capture the full conical region even when it extends beyond arena
        
        # Get coordinates of ALL pixels in the conical mask (not just non-black)
        y_coords, x_coords = np.where(conical_mask)
        
        if len(x_coords) == 0:
            # No pixels found, return empty view
            empty_view = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            return empty_view
        
        # Get the pixel values at these coordinates (including black boundary pixels)
        valid_pixels = arena_img[conical_mask]
        
        # Convert to relative coordinates (relative to robot position)
        x_rel = x_coords - center_x
        y_rel = y_coords - center_y
        
        # Convert to polar coordinates for rotation
        distances = np.sqrt(x_rel**2 + y_rel**2)
        angles = np.arctan2(-y_rel, x_rel)  # Image coordinates: y increases downward
        
        # Rotate angles to normalize orientation (forward = up)
        rotation_rad = np.deg2rad(90 - rob_yaw_deg)
        rotated_angles = angles + rotation_rad
        
        # Convert back to Cartesian coordinates (relative to center)
        x_rotated_rel = distances * np.cos(rotated_angles)
        y_rotated_rel = -distances * np.sin(rotated_angles)  # Negative for image coordinates
        
        # Convert to absolute coordinates in a large canvas
        canvas_size = 2 * radius_px  # Large enough to hold the rotated cone
        canvas_center = canvas_size // 2
        
        x_rotated_abs = x_rotated_rel + canvas_center
        y_rotated_abs = y_rotated_rel + canvas_center
        
        # Create a canvas for the rotated cone
        rotated_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        
        # Place the rotated pixels on the canvas
        valid_indices = (x_rotated_abs >= 0) & (x_rotated_abs < canvas_size) & \
                       (y_rotated_abs >= 0) & (y_rotated_abs < canvas_size)
        
        x_valid = x_rotated_abs[valid_indices].astype(int)
        y_valid = y_rotated_abs[valid_indices].astype(int)
        
        rotated_canvas[y_valid, x_valid] = valid_pixels[valid_indices]
        
        # The robot position is now at the center of the canvas
        region_center_x = canvas_center
        region_center_y = canvas_center
        
        rotated_region = rotated_canvas
        
        # After rotation, the cone tip (robot position) should already be at the rotation center
        # The rotation was performed around (region_center_x, region_center_y), which is the robot position
        # So the cone tip should be at that location. We just need to ensure this point is centered.
        
        # The cone tip is at the rotation center: (region_center_x, region_center_y)
        cone_tip_x = region_center_x
        cone_tip_y = region_center_y
        
        # Calculate translation to center the cone tip precisely
        target_center_x = rotated_region.shape[1] // 2
        target_center_y = rotated_region.shape[0] // 2
        
        translation_x = target_center_x - cone_tip_x
        translation_y = target_center_y - cone_tip_y
        
        # Apply the translation to center the cone tip precisely
        translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
        centered_region = cv2.warpAffine(
            rotated_region,
            translation_matrix,
            (rotated_region.shape[1], rotated_region.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)  # Fill with black
        )
        
        # Crop the center portion that contains the normalized, centered cone
        crop_size = min(centered_region.shape[:2])
        start_x = (centered_region.shape[1] - crop_size) // 2
        start_y = (centered_region.shape[0] - crop_size) // 2
        cropped_region = centered_region[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # Resize to target output size
        conical_view = cv2.resize(cropped_region, output_size, interpolation=cv2.INTER_AREA)
        
        if visualize:
            self._visualize_conical_extraction(
                arena_img, conical_mask, conical_view, 
                center_x, center_y, radius_px, rob_yaw_deg, opening_angle_deg,
                save_path=save_path
            )
        
        return conical_view

    def _visualize_conical_extraction(self, arena_img, conical_mask, conical_view, 
                                     center_x, center_y, radius_px, rob_yaw_deg, opening_angle_deg,
                                     save_path=None):
        """
        Visualize the conical view extraction process.
        
        Parameters
        ----------
        arena_img : numpy.ndarray
            Original arena image
        conical_mask : numpy.ndarray
            Binary mask of conical region
        conical_view : numpy.ndarray
            Extracted conical view
        center_x, center_y : int
            Center coordinates in pixels
        radius_px : int
            Radius in pixels
        rob_yaw_deg : float
            Robot orientation in degrees
        opening_angle_deg : float
            Opening angle of the cone in degrees
        save_path : str, optional
            If provided, saves the plot to this path instead of showing it
        """
        import matplotlib.pyplot as plt
        
        # Create figure with simplified 1x2 layout
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Arena with conical mask overlay (combined the two arena views)
        plt.subplot(1, 2, 1)
        plt.imshow(arena_img)
        plt.scatter(center_x, center_y, c='red', s=100, marker='x')
        
        # Draw cone outline - this must match exactly how the conical mask is created
        # Use the same angle calculation as create_conical_mask for consistency
        half_opening_rad = np.deg2rad(opening_angle_deg / 2)
        orientation_rad = np.deg2rad(rob_yaw_deg)
        
        # Calculate cone boundaries using the corrected angle (same as create_conical_mask)
        lower_bound = orientation_rad - half_opening_rad
        upper_bound = orientation_rad + half_opening_rad
        
        # Generate angles within the cone boundaries
        cone_theta = np.linspace(lower_bound, upper_bound, 100)
        
        # Convert polar coordinates to image coordinates
        # Note: We use the same coordinate system as create_conical_mask
        x_cone = center_x + radius_px * np.cos(cone_theta)
        y_cone = center_y - radius_px * np.sin(cone_theta)  # Negate y for image coordinates
        
        plt.plot(x_cone, y_cone, 'r-', linewidth=2)
        plt.plot([center_x, x_cone[0]], [center_y, y_cone[0]], 'r-', linewidth=2)
        plt.plot([center_x, x_cone[-1]], [center_y, y_cone[-1]], 'r-', linewidth=2)
        
        # Draw robot orientation arrow (using corrected coordinate system to match cone)
        arrow_length = 80
        end_x = center_x + arrow_length * np.cos(np.deg2rad(rob_yaw_deg))
        end_y = center_y - arrow_length * np.sin(np.deg2rad(rob_yaw_deg))  # Negate y for image coordinates
        plt.arrow(center_x, center_y, end_x - center_x, end_y - center_y,
                  head_width=8, head_length=12, fc='yellow', ec='yellow')
        plt.text(end_x, end_y, f'Robot orientation: {rob_yaw_deg:.1f}°',
                 color='yellow', bbox=dict(facecolor='black', alpha=0.7))
        
        # Overlay the conical mask with transparency
        mask_display = np.zeros((*arena_img.shape[:2], 3), dtype=np.uint8)
        mask_display[conical_mask] = [255, 0, 0]  # Red for cone region
        plt.imshow(mask_display, alpha=0.3)
        
        plt.title(f'Arena with Conical Region ({opening_angle_deg}° opening)')
        plt.axis('off')
        
        # Plot 2: Normalized conical view (forward direction always points UP)
        plt.subplot(1, 2, 2)
        plt.imshow(conical_view)
        
        # In the normalized view, the robot position (cone tip) should be exactly at the center
        # This is because we rotate and translate to ensure the robot is at the center
        center = (conical_view.shape[1] // 2, conical_view.shape[0] // 2)
        
        # Draw robot position (cone tip) at center (red X)
        plt.scatter(center[0], center[1], c='red', s=100, marker='x', linewidth=2)
        plt.text(center[0] + 10, center[1], f'Robot: ({center[0]}, {center[1]})',
                color='red', bbox=dict(facecolor='white', alpha=0.7))
        
        # Draw arrow from robot position upward (yellow arrow) - this shows forward direction
        arrow_length = min(conical_view.shape[:2]) // 4
        end_y = center[1] - arrow_length
        plt.arrow(center[0], center[1], 0, -arrow_length,
                 head_width=10, head_length=15, fc='yellow', ec='yellow')
        
        # Add text to clarify this is a normalized view
        plt.text(center[0], 20, 
                'Normalized Conical View', 
                color='white', 
                bbox=dict(facecolor='black', alpha=0.7),
                ha='center')
        plt.text(center[0], 40, 
                'Red X = Robot Position (should be at center)', 
                color='white', 
                bbox=dict(facecolor='black', alpha=0.7),
                ha='center')
        plt.text(center[0], 60, 
                'Yellow Arrow = Forward Direction (always UP)', 
                color='white', 
                bbox=dict(facecolor='black', alpha=0.7),
                ha='center')
        
        plt.title(f'Normalized Conical View ({conical_view.shape[1]}x{conical_view.shape[0]})')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_arena(self, show=True):
        plt.figure()
        plt.scatter(self.wall_x, self.wall_y, color='green', s=1)
        plt.scatter(self.path_x, self.path_y, color='red', s=5)
        plt.scatter(self.rob_x, self.rob_y, color='black', s=1)
        plt.xlabel('x mm')
        plt.ylabel('y mm')
        plt.axis('equal')
        if show: plt.show()

    def plot_all_sonar(self, view=False, profile=False, output_dir='Plots', dpi=150, indices=None):
        """
        Generate and save plots for all sonar measurements with optional views and profiles.
        
        Parameters
        ----------
        view : bool, optional
            If True, includes conical views in the plots (default: False)
        profile : bool, optional
            If True, includes distance profiles in the plots as polar plots (default: False)
            Polar plots provide better visualization of the geometric relationship between
            azimuth angles and distances, making it easier to understand the spatial
            distribution of obstacles around the robot.
        output_dir : str, optional
            Directory to save plots (default: 'Plots')
        dpi : int, optional
            DPI for saved images (default: 150)
        indices : range, list, or None, optional
            Specifies which indices to plot. Can be:
            - None: plot all measurements (default)
            - range(start, stop, step): plot indices from start to stop-1 with given step
            - list of integers: plot only the specified indices
            Examples:
                range(0, 10) - plot indices 0 through 9
                range(0, 10, 2) - plot indices 0, 2, 4, 6, 8
                [0, 5, 10, 15] - plot only these specific indices
            
        Notes
        -----
        - Creates a subdirectory with timestamp for organized storage
        - Uses tqdm to show progress
        - Saves plots as PNG files without displaying them
        - Automatically loads sonar data if not already loaded
        - If view=True, requires views to be loaded
        - If profile=True, requires profiles to be loaded
        """
        import os
        import time
        from datetime import datetime
        
        # Validate view requirement
        if view and (not hasattr(self, 'views') or self.views is None):
            raise ValueError("Views not loaded. Call load_views() first or set view=False.")
        
        # Validate profile requirement  
        if profile and (not hasattr(self, 'profiles') or self.profiles is None):
            raise ValueError("Profiles not loaded. Call load_profiles() first or set profile=False.")
        
        # Automatically load sonar data if not already loaded
        if not hasattr(self, 'sonar_data') or self.sonar_data is None:
            print("📡 Loading sonar data for plotting...")
            self.load_sonar()

        # Load corrected distance/IID if available
        try:
            if not hasattr(self, 'sonar_distance') or self.sonar_distance is None:
                self.sonar_distance = np.array(self.get_field('sonar_package', 'corrected_distance'))
            if not hasattr(self, 'sonar_iid') or self.sonar_iid is None:
                self.sonar_iid = np.array(self.get_field('sonar_package', 'corrected_iid'))
        except Exception:
            self.sonar_distance = None
            self.sonar_iid = None
        
        # Create output directory with timestamp
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = os.path.basename(self.session)
        plot_dir = os.path.join(output_dir, f'sonar_plots_{session_name}_{timestamp}')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Ensure sonar data is loaded
        if not hasattr(self, 'sonar_data') or self.sonar_data is None:
            self.load_sonar(flatten=False)
        
        print(f"📊 Generating sonar plots for {len(self.sonar_data)} measurements...")
        print(f"   Output directory: {plot_dir}")
        print(f"   Options: view={view}, profile={profile}")
        
        # Determine plot layout based on options
        n_subplots = 1  # Always have sonar plot
        if view:
            n_subplots += 1
        if profile:
            n_subplots += 1
        
        # Determine which indices to plot based on indices parameter
        if indices is None:
            indices_to_plot = range(len(self.sonar_data))
        elif isinstance(indices, (list, tuple)):
            # Convert list/tuple to range-like object
            indices_to_plot = indices
        elif hasattr(indices, 'start') and hasattr(indices, 'stop'):
            # It's a range object
            indices_to_plot = list(indices)
        else:
            raise ValueError(f"indices parameter must be None, list, or range object, got {type(indices)}")
        
        # Create plots for each specified measurement
        # Create plots for each specified measurement
        for index in tqdm(indices_to_plot, desc="Plotting sonar data"):
            sonar_data = self.sonar_data[index]
            rob_x = self.rob_x[index]
            rob_y = self.rob_y[index]
            rob_yaw = self.rob_yaw_deg[index]
            
            # Create figure with appropriate layout
            if n_subplots == 1:
                fig, axs = plt.subplots(1, 1, figsize=(10, 6))
                axs = [axs]  # Make it iterable
            elif n_subplots == 2:
                fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            else:  # n_subplots == 3
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            
            # Convert profile subplot to polar if needed
            if profile and n_subplots == 3:
                # For 3 subplots, profile is the last one (index 2)
                # Need to recreate the subplot with polar projection
                fig.delaxes(axs[2])
                axs[2] = fig.add_subplot(1, 3, 3, polar=True)
            elif profile and n_subplots == 2:
                # For 2 subplots, profile is the second one (index 1)
                # Need to recreate the subplot with polar projection
                fig.delaxes(axs[1])
                axs[1] = fig.add_subplot(1, 2, 2, polar=True)
            
            # Plot sonar data
            ax = axs[0]
            distance_axis = Utils.get_distance_axis(10000, sonar_data.shape[0])  # Assuming 10kHz sample rate
            
            # Plot both channels
            ax.plot(distance_axis, sonar_data[:, 0], 'b-', alpha=0.7, label='Left Channel')
            ax.plot(distance_axis, sonar_data[:, 1], 'r-', alpha=0.7, label='Right Channel')
            
            ax.set_title(f'Sonar Measurement #{index}')
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add robot position info
            pos_info = f'Pos: ({rob_x:.1f}, {rob_y:.1f}) mm\nYaw: {rob_yaw:.1f}°'
            if self.sonar_distance is not None and self.sonar_iid is not None:
                dist_val = float(self.sonar_distance[index])
                iid_val = float(self.sonar_iid[index])
                pos_info += f'\nDist: {dist_val:.3f} m\nIID: {iid_val:.3f} dB'
            ax.text(0.02, 0.95, pos_info, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot view if requested
            current_plot = 1
            if view:
                ax = axs[current_plot]
                view_img = self.views[index]
                ax.imshow(view_img)
                ax.set_title(f'Conical View #{index}')
                ax.axis('off')
                current_plot += 1
            
            # Plot profile if requested
            if profile:
                ax = axs[current_plot]
                profile_data = self.profiles[index]
                centers = self.profile_centers[index]
                
                # Convert to polar coordinates for better visualization
                # Convert azimuth angles to radians and adjust for polar plot convention
                theta = np.deg2rad(centers)
                r = profile_data
                
                # Create polar plot
                ax.set_theta_zero_location('N')  # 0° at top (forward direction)
                ax.set_theta_direction(-1)      # Clockwise direction (more intuitive)
                
                # Plot the polar profile
                ax.plot(theta, r, 'go', markersize=2)
                ax.set_title(f'Distance Profile #{index} (Polar)')
                ax.set_rlabel_position(45)  # Move radial labels away from plot
                ax.grid(True, alpha=0.3)
                
                # Add azimuth angle labels
                ax.set_xticks(np.deg2rad([0, 30, 60, 90, 120, 150, 180]))
                ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°'])
            
            # Save plot
            plot_filename = os.path.join(plot_dir, f'sonar_{index:04d}.png')
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        
        num_plots_created = len(indices_to_plot) if hasattr(indices_to_plot, '__len__') else len(self.sonar_data)
        print(f"✅ Saved {num_plots_created} sonar plots to: {plot_dir}")
        print(f"   Total files: {len(os.listdir(plot_dir))}")
        print(f"   Use: ls {plot_dir} to view files")
        
        return plot_dir

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
        if flatten: sonar_data = flatten_sonar(sonar_data)
        return sonar_data



    def get_profile(self, rob_x, rob_y, rob_yaw_deg, min_az_deg, max_az_deg, n, profile_method='min_bin'):
        """
        Get a distance profile from walls in the arena.
        
        Parameters
        ----------
        rob_x, rob_y, rob_yaw_deg : float
            Robot position and orientation
        min_az_deg, max_az_deg : float
            Azimuth range for the profile in degrees
        n : int
            Number of azimuth steps in the profile
            
        profile_method : str
            Profile extraction method:
            - 'min_bin': minimum distance among wall points within each azimuth bin.
            - 'ray_center': approximate ray-cast at each bin center.

        Returns
        -------
        centers, min_distances : numpy.ndarray
            Azimuth centers and corresponding minimum distances to walls
        """
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

        if profile_method == 'min_bin':
            for i in range(n):
                in_slice = (angles_deg >= edges[i]) & (angles_deg < edges[i + 1])
                if np.any(in_slice):
                    min_distances[i] = np.min(distances[in_slice])
        elif profile_method == 'ray_center':
            # Approximate ray-cast on point-wall map:
            # choose nearest forward point close to each center-ray.
            mm_per_px = float(self.meta.get("map_mm_per_px", 10.0))
            ray_tolerance_mm = max(1.5 * mm_per_px, 15.0)
            half_bin = 0.5 * (edges[1] - edges[0]) if n > 1 else 180.0

            for i, center_deg in enumerate(centers):
                th = np.deg2rad(center_deg)
                c = np.cos(th)
                s = np.sin(th)

                # Projection along ray (+forward only) and perpendicular distance.
                t = rel_x * c + rel_y * s
                d_perp = np.abs(-rel_x * s + rel_y * c)

                # Restrict to points with near-center azimuth to reduce accidental side hits.
                angle_diff = (angles_deg - center_deg + 180.0) % 360.0 - 180.0
                near_az = np.abs(angle_diff) <= half_bin

                candidates = (t > 0.0) & (d_perp <= ray_tolerance_mm) & near_az
                if np.any(candidates):
                    min_distances[i] = float(np.min(t[candidates]))
        else:
            raise ValueError(
                f"Unknown profile_method='{profile_method}'. "
                "Use 'min_bin' or 'ray_center'."
            )

        return centers, min_distances

    def get_profile_at(self, index, min_az_deg, max_az_deg, n, fill_nans=True, profile_method='min_bin'):
        """
        Get a distance profile for a specific robot position.
        
        Parameters
        ----------
        index : int
            Index of the robot position
        min_az_deg, max_az_deg : float
            Azimuth range for the profile in degrees
        n : int
            Number of azimuth steps in the profile
        fill_nans : bool
            If True, linearly fill NaNs in the returned profile.
        profile_method : str
            Profile extraction method. See `get_profile(...)`.
            
        Returns
        -------
        centers, profile : numpy.ndarray
            Azimuth centers and distance profile with NaN values filled
        """
        selected_x = self.rob_x[index]
        selected_y = self.rob_y[index]
        selected_yaw_deg = self.rob_yaw_deg[index]
        centers, profile = self.get_profile(
            selected_x,
            selected_y,
            selected_yaw_deg,
            min_az_deg,
            max_az_deg,
            n,
            profile_method=profile_method,
        )
        if fill_nans:
            profile = Utils.fill_nans_linear(profile)
        profile = np.asarray(profile, dtype=np.float32)
        return centers, profile

    def get_position_at(self, index):
        selected_x = self.rob_x[index]
        selected_y = self.rob_y[index]
        selected_yaw_deg = self.rob_yaw_deg[index]
        return selected_x, selected_y, selected_yaw_deg
    
    def _calculate_spatial_quadrants(self):
        """
        Calculate spatial quadrants for all robot positions.
        
        Divides the arena into 4 quadrants based on mean x and y positions:
        - Quadrant 0: x ≤ mean_x AND y ≤ mean_y (bottom-left)
        - Quadrant 1: x > mean_x AND y ≤ mean_y (bottom-right)
        - Quadrant 2: x > mean_x AND y > mean_y (top-right)
        - Quadrant 3: x ≤ mean_x AND y > mean_y (top-left)
        
        Results are stored in self.quadrants as a numpy array of integers.
        """
        # Calculate mean positions to determine quadrant boundaries
        mean_x = np.mean(self.rob_x)
        mean_y = np.mean(self.rob_y)
        
        # Initialize quadrants array
        quadrants = np.zeros(self.n, dtype=int)
        
        # Assign quadrants based on position relative to means
        # Quadrant 0: x ≤ mean_x AND y ≤ mean_y (bottom-left)
        mask_q0 = (self.rob_x <= mean_x) & (self.rob_y <= mean_y)
        quadrants[mask_q0] = 0
        
        # Quadrant 1: x > mean_x AND y ≤ mean_y (bottom-right)
        mask_q1 = (self.rob_x > mean_x) & (self.rob_y <= mean_y)
        quadrants[mask_q1] = 1
        
        # Quadrant 2: x > mean_x AND y > mean_y (top-right)
        mask_q2 = (self.rob_x > mean_x) & (self.rob_y > mean_y)
        quadrants[mask_q2] = 2
        
        # Quadrant 3: x ≤ mean_x AND y > mean_y (top-left)
        mask_q3 = (self.rob_x <= mean_x) & (self.rob_y > mean_y)
        quadrants[mask_q3] = 3
        
        self.quadrants = quadrants
        self.mean_x = mean_x
        self.mean_y = mean_y
        
        print(f"🗺️  Spatial quadrants calculated:")
        print(f"   Mean X: {mean_x:.1f} mm, Mean Y: {mean_y:.1f} mm")
        print(f"   Quadrant distribution: Q0={np.sum(quadrants==0)}, Q1={np.sum(quadrants==1)}, "
              f"Q2={np.sum(quadrants==2)}, Q3={np.sum(quadrants==3)}")




    # def get_vector_field(self, d=100, plot=False):
    #     polyline = Guidance.build_polyline_from_xy(self.path_x, self.path_y, plot=plot)
    #     rob_x = self.rob_x
    #     rob_y = self.rob_y
    #     rob_yaw_deg = self.rob_yaw_deg
    #     df = Guidance.guidance_vector_field_batch(rob_x, rob_y, rob_yaw_deg, polyline, d, visualize=plot)
    #     return df, polyline

    # def get_wall_vector_field(
    #         self,
    #         plot=False,
    #         close_iter=2,
    #         dilate_iter=2,
    #         seed_xy=None,
    # ):
    #     rob_x = self.rob_x
    #     rob_y = self.rob_y
    #     rob_yaw_deg = self.rob_yaw_deg
    #     if seed_xy is None:
    #         seed_xy = (
    #             float(rob_x[0]),
    #             float(rob_y[0]),
    #         )
    #     seed_col, seed_row = Vectors.world_to_pixel(seed_xy[0], seed_xy[1], self.meta)
    #     df, dist, inside, mask = Vectors.wall_vector_field_batch(
    #         rob_x,
    #         rob_y,
    #         rob_yaw_deg,
    #         self.wall_mask,
    #         self.meta,
    #         seed_xy=(seed_col, seed_row),
    #         close_iter=close_iter,
    #         dilate_iter=dilate_iter,
    #         visualize=plot,
    #         invert_y=False,
    #         title="Wall repulsion field (poses)",
    #     )
    #     return df, dist, inside, mask
