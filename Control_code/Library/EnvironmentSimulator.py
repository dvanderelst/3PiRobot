"""
Environment Simulator for robot navigation with sonar emulation.

This module provides tools to simulate a robot moving through an arena and
predict what sonar measurements (distance and IID) it would receive using
the trained emulator.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from Library.Emulator import Emulator
from Library import DataProcessor


class ArenaLayout:
    """
    Represents the physical layout of an arena with walls and obstacles.
    
    This class loads wall data from a session and provides methods to
    compute distance profiles from arbitrary positions.
    """
    
    def __init__(self, session_name: str):
        """
        Initialize arena layout from a session.
        
        Args:
            session_name: Name of the session to load (e.g., "sessionB01")
        """
        self.session_name = session_name
        self.dc = DataProcessor.DataCollection([session_name])
        
        # Load arena metadata through the processor
        self._load_arena_metadata()
        
        # Load wall data
        self.walls = self._load_walls()
        
        # Arena boundaries (with fallback defaults)
        self.arena_width = float(self.meta.get("arena_width_mm", 2400.0))
        self.arena_height = float(self.meta.get("arena_height_mm", 1800.0))
        self.mm_per_px = float(self.meta.get("map_mm_per_px", 10.0))
        
    def _load_arena_metadata(self):
        """Load arena metadata from the session."""
        try:
            # Access the first processor's metadata
            if self.dc.processors:
                processor = self.dc.processors[0]
                # Try to load arena metadata if not already loaded
                if not hasattr(processor, 'arena_metadata_loaded') or not processor.arena_metadata_loaded:
                    processor.load_arena_metadata()
                
                # Get metadata
                if hasattr(processor, 'meta'):
                    self.meta = processor.meta
                else:
                    # Fallback: create minimal meta dict
                    self.meta = {}
            else:
                self.meta = {}
        except Exception as e:
            print(f"⚠ Could not load arena metadata: {e}")
            print("  Using default arena parameters")
            self.meta = {}
    
    def _use_default_arena(self):
        """Set up a default arena when no session data is available."""
        self.session_name = "_default_"
        self.dc = None
        
        # Set default metadata
        self.meta = {
            "arena_width_mm": 2400.0,
            "arena_height_mm": 1800.0,
            "map_mm_per_px": 10.0
        }
        
        # Default arena boundaries
        self.arena_width = 2400.0
        self.arena_height = 1800.0
        self.mm_per_px = 10.0
        
        # Create a simple rectangular arena wall
        self.walls = np.array([
            [0, 0], [2400, 0], [2400, 1800], [0, 1800], [0, 0]  # Close the loop
        ], dtype=np.float32)
        
        print(f"✅ Created default arena: {self.arena_width}mm × {self.arena_height}mm")
    
    def _load_walls(self) -> np.ndarray:
        """Load wall coordinates from the session."""
        try:
            # Try to get walls from processor
            if self.dc.processors:
                processor = self.dc.processors[0]
                if hasattr(processor, 'wall_x') and hasattr(processor, 'wall_y'):
                    # Stack coordinates into (N, 2) array
                    walls = np.column_stack((processor.wall_x, processor.wall_y))
                    
                    # According to DataProcessor, these should already be in mm
                    # But let's verify the scale
                    max_coord = np.max(walls)
                    if max_coord > 10000:  # Probably meters if > 10,000
                        walls = walls * 1000  # Convert meters to mm
                        print(f"✓ Converted wall coordinates from meters to mm")
                    elif max_coord < 100:  # Probably pixels if < 100
                        walls = walls * self.mm_per_px
                        print(f"✓ Converted wall coordinates from pixels to mm (scale: {self.mm_per_px}mm/px)")
                    else:
                        print(f"✓ Wall coordinates appear to be in mm (max: {max_coord:.1f}mm)")
                    
                    return walls.astype(np.float32)
            
            # Fallback: try DataCollection method
            walls = self.dc.get_field("arena", "walls")
            if walls is not None:
                walls = np.array(walls, dtype=np.float32)
                return walls
        except Exception as e:
            print(f"⚠ Error loading walls: {e}")
        
        # Final fallback: empty array
        print("⚠ No wall data available - profile generation will return NaN")
        return np.array([], dtype=np.float32)
    
    def get_relative_wall_coordinates(self, rob_x: float, rob_y: float, rob_yaw_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get wall coordinates relative to robot position and orientation.
        
        Args:
            rob_x, rob_y: Robot position in mm
            rob_yaw_deg: Robot orientation in degrees
            
        Returns:
            rel_x, rel_y: Wall coordinates relative to robot
        """
        if len(self.walls) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        
        # Convert robot yaw to radians
        yaw_rad = np.deg2rad(rob_yaw_deg)
        
        # Transform wall coordinates to robot-centric frame
        # Robot position in pixels
        rob_x_px = rob_x / self.mm_per_px
        rob_y_px = rob_y / self.mm_per_px
        
        # Wall coordinates in pixels (assuming walls are in mm)
        wall_x_px = self.walls[:, 0] / self.mm_per_px
        wall_y_px = self.walls[:, 1] / self.mm_per_px
        
        # Translate to robot position
        rel_x_px = wall_x_px - rob_x_px
        rel_y_px = wall_y_px - rob_y_px
        
        # Rotate to robot orientation
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        
        rel_x_rot = rel_x_px * cos_yaw + rel_y_px * sin_yaw
        rel_y_rot = -rel_x_px * sin_yaw + rel_y_px * cos_yaw
        
        # Convert back to mm
        rel_x_mm = rel_x_rot * self.mm_per_px
        rel_y_mm = rel_y_rot * self.mm_per_px
        
        return rel_x_mm, rel_y_mm
    
    def compute_profile(self, rob_x: float, rob_y: float, rob_yaw_deg: float, 
                       min_az_deg: float, max_az_deg: float, n_steps: int,
                       profile_method: str = 'min_bin') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distance profile from robot position and orientation.
        
        Args:
            rob_x, rob_y: Robot position in mm
            rob_yaw_deg: Robot orientation in degrees
            min_az_deg, max_az_deg: Azimuth range for profile
            n_steps: Number of azimuth steps
            profile_method: 'min_bin' or 'ray_center'
            
        Returns:
            centers: Azimuth centers in degrees
            distances: Minimum distances at each azimuth in mm
        """
        # Get relative wall coordinates
        rel_x, rel_y = self.get_relative_wall_coordinates(rob_x, rob_y, rob_yaw_deg)
        
        if len(rel_x) == 0:
            # No walls available, return NaN profile
            edges = np.linspace(min_az_deg, max_az_deg, n_steps + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return centers, np.full(n_steps, np.nan, dtype=np.float32)
        
        # Calculate angles and distances of wall points relative to robot
        angles_deg = np.rad2deg(np.arctan2(rel_y, rel_x))
        distances = np.hypot(rel_x, rel_y)
        
        # Create azimuth bins
        edges = np.linspace(min_az_deg, max_az_deg, n_steps + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        min_distances = np.full(n_steps, np.nan, dtype=np.float32)
        
        if profile_method == 'min_bin':
            # Minimum distance within each azimuth bin
            for i in range(n_steps):
                in_bin = (angles_deg >= edges[i]) & (angles_deg < edges[i + 1])
                if np.any(in_bin):
                    min_distances[i] = np.min(distances[in_bin])
                    
        elif profile_method == 'ray_center':
            # Approximate ray-cast at each bin center
            mm_per_px = self.mm_per_px
            ray_tolerance_mm = max(1.5 * mm_per_px, 15.0)
            half_bin = 0.5 * (edges[1] - edges[0]) if n_steps > 1 else 180.0
            
            for i, center_deg in enumerate(centers):
                th = np.deg2rad(center_deg)
                c = np.cos(th)
                
                # Find points near this ray direction
                angle_diff = np.abs(angles_deg - center_deg)
                min_angle_diff = np.min(angle_diff)
                
                if min_angle_diff <= half_bin:
                    # Find closest points to this ray
                    on_ray = np.abs(angle_diff) <= half_bin
                    if np.any(on_ray):
                        min_distances[i] = np.min(distances[on_ray])
        
        return centers, min_distances


class EnvironmentSimulator:
    """
    Main environment simulator that combines arena layout with sonar emulator.
    
    This class provides the core simulation functionality for policy learning.
    """
    
    def __init__(self, session_name: str = "sessionB01"):
        """
        Initialize simulator with arena layout and emulator.
        
        Args:
            session_name: Session to use for arena layout
        """
        # Load arena layout
        self.arena = ArenaLayout(session_name)
        
        # Load emulator
        self.emulator = Emulator.load(device="cpu")  # Use CPU for stability
        
        # Get profile parameters from emulator (ensures consistency)
        self.profile_params = self.emulator.get_profile_params()
        self.opening_angle = self.profile_params['profile_opening_angle']
        self.profile_steps = self.profile_params['profile_steps']
        
        print(f"Simulator initialized with {session_name}")
        print(f"Profile config: {self.opening_angle}° opening, {self.profile_steps} steps")
        print(f"Arena size: {self.arena.arena_width:.0f}mm × {self.arena.arena_height:.0f}mm")
    
    def get_profile_at_position(self, x: float, y: float, orientation_deg: float) -> np.ndarray:
        """
        Get distance profile at specified position and orientation.
        
        Args:
            x, y: Position in mm
            orientation_deg: Robot orientation in degrees
            
        Returns:
            Distance profile array of shape (profile_steps,)
        """
        # Calculate azimuth range for the profile
        half_opening = self.opening_angle / 2
        min_az = -half_opening
        max_az = half_opening
        
        # Compute profile
        centers, distances = self.arena.compute_profile(
            x, y, orientation_deg,
            min_az, max_az, self.profile_steps,
            profile_method='min_bin'
        )
        
        return distances
    
    def get_sonar_measurement(self, x: float, y: float, orientation_deg: float) -> Dict[str, float]:
        """
        Get predicted sonar measurement (distance and IID) at position/orientation.
        
        This is the main interface for policy learning.
        
        Args:
            x, y: Position in mm
            orientation_deg: Robot orientation in degrees
            
        Returns:
            Dictionary with 'distance_mm' and 'iid_db'
        """
        # Get distance profile
        profile = self.get_profile_at_position(x, y, orientation_deg)
        
        # Use emulator to predict sonar measurements
        result = self.emulator.predict_single(profile)
        
        return result
    
    def simulate_robot_movement(self, start_x: float, start_y: float, start_orientation: float,
                               actions: List[Dict[str, float]]) -> List[Dict[str, Union[float, Dict]]]:
        """
        Simulate a sequence of robot movements and get sonar measurements.
        
        Args:
            start_x, start_y: Starting position in mm
            start_orientation: Starting orientation in degrees
            actions: List of action dictionaries with keys:
                    'rotate1_deg', 'rotate2_deg', 'drive_mm'
            
        Returns:
            List of states with position, orientation, and sonar measurements
        """
        trajectory = []
        current_x, current_y = start_x, start_y
        current_orientation = start_orientation
        
        for step, action in enumerate(actions):
            # Apply rotate1
            current_orientation += action['rotate1_deg']
            current_orientation = current_orientation % 360  # Normalize
            
            # Take measurement after rotate1
            measurement_after_rotate1 = self.get_sonar_measurement(
                current_x, current_y, current_orientation
            )
            
            # Apply rotate2
            current_orientation += action['rotate2_deg']
            current_orientation = current_orientation % 360  # Normalize
            
            # Take measurement after rotate2 (before driving)
            measurement_after_rotate2 = self.get_sonar_measurement(
                current_x, current_y, current_orientation
            )
            
            # Apply drive
            drive_distance = action['drive_mm']
            drive_rad = np.deg2rad(current_orientation)
            
            current_x += drive_distance * np.cos(drive_rad)
            current_y += drive_distance * np.sin(drive_rad)
            
            # Ensure robot stays within arena bounds
            current_x = np.clip(current_x, 50, self.arena.arena_width - 50)
            current_y = np.clip(current_y, 50, self.arena.arena_height - 50)
            
            # Take measurement after driving
            measurement_after_drive = self.get_sonar_measurement(
                current_x, current_y, current_orientation
            )
            
            # Store state
            state = {
                'step': step,
                'position': {'x': current_x, 'y': current_y},
                'orientation': current_orientation,
                'actions': action,
                'measurements': {
                    'after_rotate1': measurement_after_rotate1,
                    'after_rotate2': measurement_after_rotate2,
                    'after_drive': measurement_after_drive
                }
            }
            trajectory.append(state)
        
        return trajectory
    
    def get_arena_info(self) -> Dict[str, Union[float, int]]:
        """Get information about the arena."""
        return {
            'width_mm': self.arena.arena_width,
            'height_mm': self.arena.arena_height,
            'mm_per_px': self.arena.mm_per_px,
            'profile_opening_angle': self.opening_angle,
            'profile_steps': self.profile_steps
        }


def create_test_simulator() -> EnvironmentSimulator:
    """Create a simulator for testing with a default session."""
    session_names = ["sessionB01", "sessionB02", "sessionB03", "sessionB04", "sessionB05"]
    
    for session_name in session_names:
        try:
            print(f"Trying to create simulator with {session_name}...")
            return EnvironmentSimulator(session_name)
        except Exception as e:
            print(f"Could not create simulator with {session_name}: {e}")
    
    # Try to find available sessions
    try:
        dc = DataProcessor.DataCollection([])
        if hasattr(dc, 'available_sessions') and dc.available_sessions:
            print(f"Available sessions: {dc.available_sessions}")
            for session_name in dc.available_sessions:
                try:
                    return EnvironmentSimulator(session_name)
                except Exception as e:
                    print(f"Could not create simulator with {session_name}: {e}")
    except Exception as e2:
        print(f"Could not check available sessions: {e2}")
    
    # Final fallback: create simulator with minimal functionality
    print("⚠ No valid sessions found - creating simulator with default arena")
    return EnvironmentSimulator("_default_")

