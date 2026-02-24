#!/usr/bin/env python3
"""
Test script for the Environment Simulator.

This script demonstrates how to use the simulator to generate profiles
and sonar measurements for arbitrary positions and orientations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from Library.EnvironmentSimulator import EnvironmentSimulator, create_test_simulator

# ============================================
# CONFIGURATION SETTINGS (Easily adjustable)
# ============================================

# Simulation session to use (try different sessions if available)
SIMULATION_SESSION = "sessionB05"  # or try "sessionB02", "sessionB03", etc.

# Number of random positions to generate for visualization
NUM_POSITIONS = 200  # More positions = denser coverage, but slower

# Visualization settings
DOT_SIZE = 100  # Size of position dots (smaller = less clutter)
MIN_ARROW_LENGTH = 20  # Minimum arrow length for distance visualization
MAX_ARROW_LENGTH = 200  # Maximum arrow length for distance visualization

# Orientation control: Set to None for random orientations, or specify a fixed yaw in degrees
FIXED_ORIENTATION = 0  # e.g., 0, 90, 180, 270, or None for random

# Output file names (saved in Plots/ folder)
DIAGNOSTIC_PLOT_FILE = "Plots/arena_diagnostic_plot.png"
UNIFIED_VISUALIZATION_FILE = "Plots/unified_arena_visualization.png"

# ============================================
# END OF CONFIGURATION
# ============================================


def test_basic_functionality():
    """Test basic simulator functionality."""
    print("Testing Environment Simulator")
    print("=" * 50)
    
    try:
        # Create simulator with configured session
        print("Creating simulator...")
        try:
            simulator = EnvironmentSimulator(SIMULATION_SESSION)
            print(f"✓ Simulator created successfully with session: {SIMULATION_SESSION}")
        except Exception as e:
            print(f"⚠ Could not create simulator with {SIMULATION_SESSION}, trying fallback...")
            simulator = create_test_simulator()
            if simulator:
                print("✓ Simulator created successfully with fallback session")
        
        # Get arena info
        arena_info = simulator.get_arena_info()
        print(f"✓ Arena info: {arena_info['width_mm']:.0f}mm × {arena_info['height_mm']:.0f}mm")
        
        return simulator
        
    except Exception as e:
        print(f"✗ Failed to create simulator: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_profile_generation(simulator: EnvironmentSimulator):
    """Test profile generation at different positions."""
    print("\nTesting Profile Generation")
    print("-" * 30)
    
    if simulator is None:
        print("✗ No simulator available")
        return
    
    try:
        # Test positions around the arena
        test_positions = [
            {"name": "Center", "x": 1200, "y": 900, "orientation": 0},
            {"name": "Corner", "x": 200, "y": 200, "orientation": 45},
            {"name": "Edge", "x": 1200, "y": 150, "orientation": 90},
            {"name": "Opposite", "x": 2200, "y": 900, "orientation": 180},
        ]
        
        for pos in test_positions:
            print(f"\nTesting {pos['name']} position:")
            print(f"  Position: ({pos['x']:.0f}, {pos['y']:.0f}mm)")
            print(f"  Orientation: {pos['orientation']}°")
            
            # Get profile
            profile = simulator.get_profile_at_position(
                pos['x'], pos['y'], pos['orientation']
            )
            
            print(f"  Profile shape: {profile.shape}")
            print(f"  Distance range: {np.nanmin(profile):.1f}-{np.nanmax(profile):.1f}mm")
            print(f"  Valid values: {np.sum(np.isfinite(profile))}/{len(profile)}")
            
            # Get sonar measurement
            measurement = simulator.get_sonar_measurement(
                pos['x'], pos['y'], pos['orientation']
            )
            
            print(f"  Sonar prediction: {measurement['distance_mm']:.1f}mm, {measurement['iid_db']:.1f}dB")
        
        print("\n✓ Profile generation tests completed")
        
    except Exception as e:
        print(f"✗ Profile generation failed: {e}")
        import traceback
        traceback.print_exc()


def test_robot_movement_simulation(simulator: EnvironmentSimulator):
    """Test simulation of robot movement sequences."""
    print("\nTesting Robot Movement Simulation")
    print("-" * 40)
    
    if simulator is None:
        print("✗ No simulator available")
        return
    
    try:
        # Define a simple movement sequence
        actions = [
            # Action 1: Look around, then drive forward
            {
                'rotate1_deg': 30,   # Look right
                'rotate2_deg': -30,  # Look forward again
                'drive_mm': 200      # Drive forward 20cm
            },
            # Action 2: Look left, then turn and drive
            {
                'rotate1_deg': -45,  # Look left
                'rotate2_deg': 45,   # Look forward
                'drive_mm': 150      # Drive forward 15cm
            },
            # Action 3: Scan, then drive
            {
                'rotate1_deg': 60,   # Look far right
                'rotate2_deg': -60,  # Look forward
                'drive_mm': 180      # Drive forward 18cm
            }
        ]
        
        # Starting position
        start_x, start_y = 1200, 500  # Near center but lower
        start_orientation = 0         # Facing "north"
        
        print(f"Starting simulation at ({start_x:.0f}, {start_y:.0f}mm), orientation {start_orientation}°")
        print(f"Action sequence: {len(actions)} steps")
        
        # Simulate movement
        trajectory = simulator.simulate_robot_movement(
            start_x, start_y, start_orientation, actions
        )
        
        print(f"\n✓ Simulation completed: {len(trajectory)} states recorded")
        
        # Print summary
        for i, state in enumerate(trajectory):
            pos = state['position']
            orient = state['orientation']
            meas_after_drive = state['measurements']['after_drive']
            
            print(f"\nStep {i+1}:")
            print(f"  Position: ({pos['x']:.0f}, {pos['y']:.0f}mm)")
            print(f"  Orientation: {orient:.1f}°")
            print(f"  Measurement: {meas_after_drive['distance_mm']:.1f}mm, {meas_after_drive['iid_db']:.1f}dB")
            
            # Show the active sensing measurements
            meas1 = state['measurements']['after_rotate1']
            meas2 = state['measurements']['after_rotate2']
            print(f"  Active sensing:")
            print(f"    After rotate1: {meas1['distance_mm']:.1f}mm, {meas1['iid_db']:.1f}dB")
            print(f"    After rotate2: {meas2['distance_mm']:.1f}mm, {meas2['iid_db']:.1f}dB")
        
        print("\n✓ Movement simulation tests completed")
        
    except Exception as e:
        print(f"✗ Movement simulation failed: {e}")
        import traceback
        traceback.print_exc()


def test_arena_coverage(simulator: EnvironmentSimulator):
    """Test measurements at a grid of positions to see arena coverage."""
    print("\nTesting Arena Coverage")
    print("-" * 25)
    
    if simulator is None:
        print("✗ No simulator available")
        return
    
    try:
        # Create a grid of test positions
        x_positions = [400, 1200, 2000]  # Left, center, right
        y_positions = [300, 900, 1500]  # Bottom, center, top
        orientations = [0, 90, 180, 270]  # Four cardinal directions
        
        print("Testing measurement coverage across arena...")
        
        results = []
        for x in x_positions:
            for y in y_positions:
                for orient in orientations:
                    try:
                        measurement = simulator.get_sonar_measurement(x, y, orient)
                        if np.isfinite(measurement['distance_mm']):
                            results.append({
                                'x': x, 'y': y, 'orient': orient,
                                'distance': measurement['distance_mm'],
                                'iid': measurement['iid_db']
                            })
                    except:
                        pass
        
        print(f"✓ Collected {len(results)} valid measurements")
        
        if results:
            distances = [r['distance'] for r in results]
            iids = [r['iid'] for r in results]
            
            print(f"  Distance range: {min(distances):.1f}-{max(distances):.1f}mm")
            print(f"  IID range: {min(iids):.1f}-{max(iids):.1f}dB")
            print(f"  Average distance: {np.mean(distances):.1f}mm")
            print(f"  Average |IID|: {np.mean(np.abs(iids)):.1f}dB")
        
        print("✓ Arena coverage test completed")
        
        return results
        
    except Exception as e:
        print(f"✗ Arena coverage test failed: {e}")
        return []


def generate_random_positions_in_arena(simulator: EnvironmentSimulator, n_positions: int = 100) -> List[Dict]:
    """
    Generate random positions within the arena polygon.
    
    Args:
        simulator: Environment simulator
        n_positions: Number of random positions to generate
        
    Returns:
        List of dictionaries with x, y, orientation
    """
    # Get arena info
    arena_info = simulator.get_arena_info()
    width = arena_info['width_mm']
    height = arena_info['height_mm']
    
    # Get wall data to create polygon
    walls = simulator.arena.walls
    if len(walls) == 0:
        # Fallback to simple rectangle if no wall data
        print("⚠ No wall data - using rectangular arena approximation")
        positions = []
        for i in range(n_positions):
            x = np.random.uniform(100, width - 100)
            y = np.random.uniform(100, height - 100)
            orient = np.random.uniform(0, 360)
            positions.append({'x': x, 'y': y, 'orient': orient})
        return positions
    
    # Create polygon from walls for point-in-polygon testing
    from matplotlib.path import Path
    wall_path = Path(walls)
    
    # Get the actual bounds of the arena from wall data
    if len(walls) > 0:
        x_min, x_max = np.min(walls[:, 0]), np.max(walls[:, 0])
        y_min, y_max = np.min(walls[:, 1]), np.max(walls[:, 1])
        # Add some padding
        x_min, x_max = x_min - 50, x_max + 50
        y_min, y_max = y_min - 50, y_max + 50
    else:
        # Fallback to expected arena size
        x_min, x_max = 0, width
        y_min, y_max = 0, height
    
    positions = []
    attempts = 0
    max_attempts = n_positions * 10  # Try harder to find valid points
    
    while len(positions) < n_positions and attempts < max_attempts:
        attempts += 1
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        orient = np.random.uniform(0, 360)
        
        # Check if point is inside arena polygon
        if wall_path.contains_point((x, y)):
            positions.append({'x': x, 'y': y, 'orient': orient})
    
    print(f"✓ Generated {len(positions)}/{n_positions} valid positions within arena")
    return positions


def create_arena_diagnostic_plot(simulator: EnvironmentSimulator) -> None:
    """
    Create diagnostic plot showing just the arena walls for verification.
    Uses similar style to SCRIPT_PlotSession.py for consistency.
    
    Args:
        simulator: Environment simulator
    """
    print("\nCreating Arena Diagnostic Plot")
    print("-" * 30)
    
    try:
        # Get arena info
        arena_info = simulator.get_arena_info()
        width = arena_info['width_mm']
        height = arena_info['height_mm']
        
        # Create figure
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Plot arena walls - use simple scatter plot like SCRIPT_PlotSession.py
        walls = simulator.arena.walls
        print(f"Wall data shape: {walls.shape}")
        print(f"Wall coordinate range: X={np.min(walls[:,0]):.1f}-{np.max(walls[:,0]):.1f}, Y={np.min(walls[:,1]):.1f}-{np.max(walls[:,1]):.1f}")
        
        if len(walls) > 0:
            # Simple scatter plot of wall points (like the original script)
            ax.scatter(walls[:, 0], walls[:, 1], color='green', s=10, alpha=0.5, label='Walls')
        else:
            print("⚠ No wall data available")
            # Plot expected arena boundaries
            expected_walls = np.array([
                [0, 0], [width, 0], [width, height], [0, height], [0, 0]
            ])
            ax.scatter(expected_walls[:, 0], expected_walls[:, 1], color='orange', s=10, alpha=0.5, label='Expected Walls')
            ax.plot(expected_walls[:, 0], expected_walls[:, 1], color='orange', linewidth=1, alpha=0.3)
        
        # Set up plot similar to SCRIPT_PlotSession.py
        ax.set_title('Arena Wall Diagnostic Plot (style matching SCRIPT_PlotSession.py)', fontsize=14, pad=15)
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        
        # Set limits based on actual wall data
        if len(walls) > 0:
            buffer = 100  # Small buffer
            x_min, x_max = np.min(walls[:, 0]) - buffer, np.max(walls[:, 0]) + buffer
            y_min, y_max = np.min(walls[:, 1]) - buffer, np.max(walls[:, 1]) + buffer
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add diagnostic info
        diagnostic_text = (
            f'Arena Wall Diagnostic\n'
            f'Expected: {width:.0f} × {height:.0f} mm\n'
            f'Wall points: {len(walls)}\n'
            f'X: {np.min(walls[:,0]):.1f}-{np.max(walls[:,0]):.1f} mm\n'
            f'Y: {np.min(walls[:,1]):.1f}-{np.max(walls[:,1]):.1f} mm'
        )
        ax.text(0.02, 0.98, diagnostic_text, transform=ax.transAxes, 
               va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save diagnostic plot using configured filename
        plt.savefig(DIAGNOSTIC_PLOT_FILE, dpi=150, bbox_inches='tight')
        print(f"✓ Diagnostic plot saved to {DIAGNOSTIC_PLOT_FILE}")
        print("  This plot uses the same style as SCRIPT_PlotSession.py for easy comparison")
        
        plt.close()
        
    except Exception as e:
        print(f"✗ Diagnostic plot failed: {e}")
        import traceback
        traceback.print_exc()


def create_integrated_arena_visualization(simulator: EnvironmentSimulator) -> None:
    """
    Create integrated visualization showing position, orientation, IID, and distance.
    
    Args:
        simulator: Environment simulator
    """
    print("\nCreating Integrated Arena Visualization")
    print("-" * 40)
    
    try:
        # Generate random positions within arena
        positions = generate_random_positions_in_arena(simulator, n_positions=100)
        
        if not positions:
            print("⚠ No valid positions generated")
            return
        
        # Get measurements for all positions
        results = []
        for i, pos in enumerate(positions):
            try:
                measurement = simulator.get_sonar_measurement(pos['x'], pos['y'], pos['orient'])
                if np.isfinite(measurement['distance_mm']):
                    results.append({
                        'x': pos['x'], 'y': pos['y'], 'orient': pos['orient'],
                        'distance': measurement['distance_mm'],
                        'iid': measurement['iid_db']
                    })
            except Exception as e:
                print(f"⚠ Measurement failed for position {i}: {e}")
        
        print(f"✓ Collected {len(results)} valid measurements")
        
        if not results:
            print("⚠ No valid measurements to visualize")
            return
        
        # Create improved visualization with separate subplots
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1], 
                             wspace=0.3, hspace=0.3)
        
        # Main plot (top-left)
        ax_main = fig.add_subplot(gs[0, 0])
        
        # IID colorbar (top-right)
        ax_iid_cbar = fig.add_subplot(gs[0, 1])
        
        # Distance colorbar (bottom-right)
        ax_dist_cbar = fig.add_subplot(gs[1, 1])
        
        # Get arena info
        arena_info = simulator.get_arena_info()
        width = arena_info['width_mm']
        height = arena_info['height_mm']
        
        # Plot actual arena walls
        walls = simulator.arena.walls
        if len(walls) > 0:
            ax_main.scatter(walls[:, 0], walls[:, 1], color='green', s=10, alpha=0.3, label='Arena Walls')
        else:
            # Fallback rectangle if no wall data
            expected_walls = np.array([
                [0, 0], [width, 0], [width, height], [0, height], [0, 0]
            ])
            ax_main.scatter(expected_walls[:, 0], expected_walls[:, 1], color='orange', s=10, alpha=0.3, label='Expected Walls')
        
        # Set up main plot
        ax_main.set_title('Sonar Measurement Distribution in Arena', fontsize=16, pad=20)
        ax_main.set_xlabel('X Position (mm)', fontsize=14)
        ax_main.set_ylabel('Y Position (mm)', fontsize=14)
        ax_main.grid(True, alpha=0.2)
        
        # Set limits
        if len(walls) > 0:
            buffer = 100
            x_min, x_max = np.min(walls[:, 0]) - buffer, np.max(walls[:, 0]) + buffer
            y_min, y_max = np.min(walls[:, 1]) - buffer, np.max(walls[:, 1]) + buffer
            ax_main.set_xlim(x_min, x_max)
            ax_main.set_ylim(y_min, y_max)
        else:
            ax_main.set_xlim(0, width)
            ax_main.set_ylim(0, height)
        
        ax_main.set_aspect('equal')
        
        # Extract data
        x_coords = [r['x'] for r in results]
        y_coords = [r['y'] for r in results]
        orientations = [r['orient'] for r in results]
        iids = [r['iid'] for r in results]
        distances = [r['distance'] for r in results]
        
        # Create colormaps with improved contrast
        iid_norm = Normalize(vmin=-10, vmax=10)  # Fixed range for consistency
        iid_cmap = plt.cm.coolwarm
        iid_mappable = ScalarMappable(norm=iid_norm, cmap=iid_cmap)
        
        dist_norm = Normalize(vmin=min(distances), vmax=max(distances))
        dist_cmap = plt.cm.plasma  # Better contrast: yellow (close) to dark blue (far)
        dist_mappable = ScalarMappable(norm=dist_norm, cmap=dist_cmap)
        
        # Plot measurements with improved visual design
        for x, y, orient, iid, dist in zip(x_coords, y_coords, orientations, iids, distances):
            # Color dot by IID (larger size, better visibility)
            dot_color = iid_cmap(iid_norm(iid))
            ax_main.scatter(x, y, color=dot_color, s=250, edgecolor='black', 
                          linewidth=1.5, alpha=0.9, zorder=3)
            
            # Variable arrow length based on distance (longer arrow = greater distance)
            min_arrow_len, max_arrow_len = 40, 120
            arrow_length = min_arrow_len + (max_arrow_len - min_arrow_len) * dist_norm(dist)
            
            # Color arrow by distance
            arrow_color = dist_cmap(dist_norm(dist))
            
            # Main arrow line (thicker, more visible)
            arrow_end_x = x + arrow_length * np.cos(np.deg2rad(orient))
            arrow_end_y = y + arrow_length * np.sin(np.deg2rad(orient))
            ax_main.plot([x, arrow_end_x], [y, arrow_end_y], 
                        color=arrow_color, linewidth=4, alpha=0.9, zorder=2)
            
            # Improved arrow head (larger, more visible)
            ax_main.arrow(x, y, 
                        (arrow_end_x - x) * 0.7, 
                        (arrow_end_y - y) * 0.7,
                        head_width=25, head_length=30, 
                        fc=arrow_color, ec='black', 
                        linewidth=1.5, alpha=0.95, zorder=4)
        
        # Add IID colorbar (top-right subplot)
        cbar_iid = plt.colorbar(iid_mappable, cax=ax_iid_cbar, orientation='vertical')
        cbar_iid.set_label('IID (dB)', fontsize=14, rotation=270, labelpad=20)
        ax_iid_cbar.set_title('Interaural\nIntensity\nDifference', fontsize=12, pad=10)
        
        # Add distance colorbar (bottom-right subplot)
        cbar_dist = plt.colorbar(dist_mappable, cax=ax_dist_cbar, orientation='vertical')
        cbar_dist.set_label('Distance (mm)', fontsize=14, rotation=270, labelpad=20)
        ax_dist_cbar.set_title('Distance\nto Obstacle', fontsize=12, pad=10)
        
        # Add comprehensive legend in main plot
        legend_text = (
            'Visualization Key:\n'
            '• Position: Dot location (colored by IID)\n'
            '• Orientation: Arrow direction (colored by distance)\n'
            '• IID: Blue=left ear louder, Red=right ear louder\n'
            '• Distance: Yellow=close, Dark blue=far\n'
            '• Arrow length: Proportional to distance (longer=farther)\n'
            f'• IID Range: {min(iids):.1f} to {max(iids):.1f} dB\n'
            f'• Distance Range: {min(distances):.0f} to {max(distances):.0f} mm\n'
            f'• Data Points: {len(results)} valid measurements'
        )
        
        # Place legend in bottom-left corner with better styling
        ax_main.text(0.02, 0.02, legend_text,
                    transform=ax_main.transAxes,
                    bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', boxstyle='round,pad=0.5'),
                    fontsize=12, va='bottom')
        
        # Add title annotation at top
        ax_main.annotate('3Pi Robot Sonar Measurement Visualization',
                        xy=(0.5, 0.98), xycoords='axes fraction',
                        ha='center', va='top',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
                        fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        output_file = 'integrated_arena_visualization.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"✓ Integrated visualization saved to {output_file}")
        
        if not os.environ.get("DISPLAY"):
            print("📊 Display not available - visualization saved to file")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"✗ Integrated visualization failed: {e}")
        import traceback
        traceback.print_exc()


def create_unified_visualization(simulator: EnvironmentSimulator) -> None:
    """
    Create unified visualization showing both IID and distance in a single panel.
    
    Args:
        simulator: Environment simulator
    """
    print("\nCreating Unified Visualization")
    print("-" * 30)
    
    try:
        # Generate random positions within arena using configured count
        positions = generate_random_positions_in_arena(simulator, n_positions=NUM_POSITIONS)
        
        if not positions:
            print("⚠ No valid positions generated")
            return
        
        # Get measurements for all positions
        results = []
        for i, pos in enumerate(positions):
            try:
                # Use fixed orientation if specified, otherwise use random orientation from position
                measurement_orient = FIXED_ORIENTATION if FIXED_ORIENTATION is not None else pos['orient']
                
                measurement = simulator.get_sonar_measurement(pos['x'], pos['y'], measurement_orient)
                if np.isfinite(measurement['distance_mm']):
                    results.append({
                        'x': pos['x'], 'y': pos['y'], 'orient': measurement_orient,
                        'distance': measurement['distance_mm'],
                        'iid': measurement['iid_db']
                    })
            except Exception as e:
                print(f"⚠ Measurement failed for position {i}: {e}")
        
        print(f"✓ Collected {len(results)} valid measurements")
        
        if not results:
            print("⚠ No valid measurements to visualize")
            return
        
        # Create single unified visualization
        fig, ax = plt.subplots(figsize=(24, 16))
        
        # Extract data
        x_coords = [r['x'] for r in results]
        y_coords = [r['y'] for r in results]
        orientations = [r['orient'] for r in results]
        iids = [r['iid'] for r in results]
        distances = [r['distance'] for r in results]
        
        # Get arena info
        arena_info = simulator.get_arena_info()
        width = arena_info['width_mm']
        height = arena_info['height_mm']
        
        # Plot arena walls
        walls = simulator.arena.walls
        if len(walls) > 0:
            ax.scatter(walls[:, 0], walls[:, 1], color='green', s=10, alpha=0.3, label='Arena Walls')
        else:
            expected_walls = np.array([
                [0, 0], [width, 0], [width, height], [0, height], [0, 0]
            ])
            ax.scatter(expected_walls[:, 0], expected_walls[:, 1], color='orange', s=10, alpha=0.3, label='Expected Walls')
        
        # Set up plot (no main title, just axis labels)
        ax.set_xlabel('X Position (mm)', fontsize=16)
        ax.set_ylabel('Y Position (mm)', fontsize=16)
        
        # Set limits and aspect
        if len(walls) > 0:
            buffer = 100
            x_min, x_max = np.min(walls[:, 0]) - buffer, np.max(walls[:, 0]) + buffer
            y_min, y_max = np.min(walls[:, 1]) - buffer, np.max(walls[:, 1]) + buffer
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Create colormaps
        iid_norm = Normalize(vmin=-10, vmax=10)
        iid_cmap = plt.cm.coolwarm
        iid_mappable = ScalarMappable(norm=iid_norm, cmap=iid_cmap)
        
        dist_norm = Normalize(vmin=min(distances), vmax=max(distances))
        dist_cmap = plt.cm.plasma
        dist_mappable = ScalarMappable(norm=dist_norm, cmap=dist_cmap)
        
        # Plot unified visualization: small IID-colored dots + distance-colored arrows
        for x, y, orient, iid, dist in zip(x_coords, y_coords, orientations, iids, distances):
            # Small dot colored by IID (configurable size)
            dot_color = iid_cmap(iid_norm(iid))
            ax.scatter(x, y, color=dot_color, s=DOT_SIZE, edgecolor='black', 
                     linewidth=1, alpha=0.8, zorder=2)
            
            # Variable arrow length based on distance (longer arrow = greater distance)
            arrow_length = MIN_ARROW_LENGTH + (MAX_ARROW_LENGTH - MIN_ARROW_LENGTH) * dist_norm(dist)
            
            # Color arrow by distance
            arrow_color = dist_cmap(dist_norm(dist))
            
            # Plot arrow
            arrow_end_x = x + arrow_length * np.cos(np.deg2rad(orient))
            arrow_end_y = y + arrow_length * np.sin(np.deg2rad(orient))
            ax.plot([x, arrow_end_x], [y, arrow_end_y], 
                   color=arrow_color, linewidth=3, alpha=0.8, zorder=3)
            
            # Arrow head
            ax.arrow(x, y, 
                    (arrow_end_x - x) * 0.8, 
                    (arrow_end_y - y) * 0.8,
                    head_width=15, head_length=20, 
                    fc=arrow_color, ec='black', 
                    linewidth=1, alpha=0.9, zorder=4)
        
        # Add IID colorbar (left side)
        cbar_iid = plt.colorbar(iid_mappable, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
        cbar_iid.set_label('IID (dB)', fontsize=14, rotation=270, labelpad=20)
        
        # Add distance colorbar (right side)
        cbar_dist = plt.colorbar(dist_mappable, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
        cbar_dist.set_label('Distance (mm)', fontsize=14, rotation=270, labelpad=20)
        
        # Add comprehensive legend
        orient_desc = "Random" if FIXED_ORIENTATION is None else f"Fixed at {FIXED_ORIENTATION}°"
        legend_text = (
            'Unified Visualization Key:\n'
            '• Small dots: Robot positions (colored by IID)\n'
            '• Arrows: Robot orientation (colored by distance)\n'
            '• IID: Blue=left ear louder, Red=right ear louder\n'
            '• Distance: Yellow=close, Dark blue=far\n'
            '• Arrow length: Proportional to distance (longer=farther)\n'
            f'• Orientation: {orient_desc}\n'
            f'• IID Range: {min(iids):.1f} to {max(iids):.1f} dB\n'
            f'• Distance Range: {min(distances):.0f} to {max(distances):.0f} mm\n'
            f'• Data Points: {len(results)} valid measurements from {len(positions)} positions'
        )
        
        # Place legend in bottom-left corner
        ax.text(0.02, 0.02, legend_text,
               transform=ax.transAxes,
               bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', boxstyle='round,pad=0.5'),
               fontsize=12, va='bottom')
        
        # No title annotation (removed for cleaner look)
        
        # Use tight_layout to ensure everything fits
        plt.tight_layout()
        
        # Save visualization using configured filename
        plt.savefig(UNIFIED_VISUALIZATION_FILE, dpi=200, bbox_inches='tight')
        print(f"✓ Unified visualization saved to {UNIFIED_VISUALIZATION_FILE}")
        
        if not os.environ.get("DISPLAY"):
            print("📊 Display not available - visualization saved to file")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        print(f"✗ Unified visualization failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all simulator tests."""
    print("Environment Simulator Test Script")
    print("=" * 50)
    
    # Create Plots directory if it doesn't exist
    try:
        plots_dir = os.path.abspath("Plots")
        os.makedirs(plots_dir, exist_ok=True)
        print(f"✓ Plots directory ready: {plots_dir}")
        # Verify directory was created
        if not os.path.exists(plots_dir):
            raise FileNotFoundError(f"Directory {plots_dir} was not created")
    except Exception as e:
        print(f"⚠ Could not create Plots directory: {e}")
        print("  Will save plots in current directory instead")
    
    orient_config = f"Random" if FIXED_ORIENTATION is None else f"Fixed ({FIXED_ORIENTATION}°)"
    print(f"Configuration: Session={SIMULATION_SESSION}, Positions={NUM_POSITIONS}, Orientation={orient_config}")
    
    # Test basic functionality
    simulator = test_basic_functionality()
    
    if simulator is not None:
        # Run additional tests
        test_profile_generation(simulator)
        test_robot_movement_simulation(simulator)
        coverage_results = test_arena_coverage(simulator)
        
        # Create diagnostic plot first
        create_arena_diagnostic_plot(simulator)
        
        # Create unified visualization only
        create_unified_visualization(simulator)
        
        print("\n" + "=" * 50)
        print("✓ All simulator tests completed successfully!")
        print("\nThe simulator is ready for policy learning.")
        print("📊 Check these files for visualization:")
        print("   - 'arena_diagnostic_plot.png' (arena walls only)")
        print("   - 'unified_arena_visualization.png' (combined IID+distance view)")
        
    else:
        print("\n✗ Simulator tests failed")


if __name__ == "__main__":
    main()