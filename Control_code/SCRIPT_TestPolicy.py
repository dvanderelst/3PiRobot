#!/usr/bin/env python3
"""
Test script for policy learning components.

This script tests the basic functionality of the policy network and environment interaction.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict

# Add Library to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Library'))

from Library.EnvironmentSimulator import EnvironmentSimulator, create_test_simulator
from Library import Utils

# Ensure reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class SimplePolicy(nn.Module):
    """Simple policy for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 3)  # Simple linear policy
        
    def forward(self, x: torch.Tensor) -> Dict[str, float]:
        """Forward pass."""
        actions = self.linear(x)
        
        # Scale actions
        rotate1 = torch.tanh(actions[0]) * 90.0  # [-90°, 90°]
        rotate2 = torch.tanh(actions[1]) * 90.0  # [-90°, 90°]
        drive = torch.sigmoid(actions[2]) * 200.0  # [0, 200mm]
        
        return {
            'rotate1_deg': rotate1.item(),
            'rotate2_deg': rotate2.item(),
            'drive_mm': drive.item()
        }

def test_environment_simulator():
    """Test the environment simulator."""
    print("Testing Environment Simulator...")
    
    try:
        simulator = create_test_simulator()
        print("✓ Simulator created successfully")
        
        # Test basic functionality
        arena_info = simulator.get_arena_info()
        print(f"  Arena size: {arena_info['width_mm']}mm × {arena_info['height_mm']}mm")
        
        # Test measurement at center
        center_x = arena_info['width_mm'] / 2
        center_y = arena_info['height_mm'] / 2
        measurement = simulator.get_sonar_measurement(center_x, center_y, 0.0)
        print(f"  Center measurement: distance={measurement['distance_mm']:.1f}mm, iid={measurement['iid_db']:.1f}dB")
        
        # Test simple movement
        start_x, start_y = 500, 500
        action = {
            'rotate1_deg': 30.0,
            'rotate2_deg': -15.0,
            'drive_mm': 100.0
        }
        trajectory = simulator.simulate_robot_movement(start_x, start_y, 0.0, [action])
        print(f"  Movement test: {len(trajectory)} steps simulated")
        final_pos = trajectory[0]['position']
        print(f"    Start: ({start_x:.1f}, {start_y:.1f})")
        print(f"    End: ({final_pos['x']:.1f}, {final_pos['y']:.1f})")
        
        return simulator
        
    except Exception as e:
        print(f"✗ Environment simulator test failed: {e}")
        return None

def test_simple_policy():
    """Test a simple policy in the environment."""
    print("\nTesting Simple Policy...")
    
    # Create simulator
    simulator = test_environment_simulator()
    if simulator is None:
        return
    
    # Create simple policy
    policy = SimplePolicy()
    print("✓ Simple policy created")
    
    # Test policy execution
    arena_info = simulator.get_arena_info()
    start_x = random.uniform(100, arena_info['width_mm'] - 100)
    start_y = random.uniform(100, arena_info['height_mm'] - 100)
    start_orientation = random.uniform(0, 360)
    
    print(f"  Starting at: ({start_x:.1f}, {start_y:.1f}), orientation: {start_orientation:.1f}°")
    
    current_x, current_y = start_x, start_y
    current_orientation = start_orientation
    
    # Run for a few steps
    for step in range(10):
        # Get measurement
        measurement = simulator.get_sonar_measurement(current_x, current_y, current_orientation)
        
        # Prepare input for policy (simple version)
        policy_input = torch.tensor([
            measurement['distance_mm'] / 2000.0,  # Normalized distance
            measurement['iid_db'] / 20.0,         # Normalized IID
            current_orientation / 360.0,         # Normalized orientation
            step / 10.0,                         # Time step
            1.0                                  # Bias
        ], dtype=torch.float32)
        
        # Get action
        action = policy(policy_input)
        
        # Execute action
        action_dict = {
            'rotate1_deg': action['rotate1_deg'],
            'rotate2_deg': action['rotate2_deg'],
            'drive_mm': action['drive_mm']
        }
        
        trajectory = simulator.simulate_robot_movement(
            current_x, current_y, current_orientation, [action_dict]
        )
        
        # Update state
        current_x = trajectory[0]['position']['x']
        current_y = trajectory[0]['position']['y']
        current_orientation = trajectory[0]['orientation']
        
        print(f"    Step {step}: Pos({current_x:.1f}, {current_y:.1f}), "
              f"Orient({current_orientation:.1f}°), "
              f"Dist({measurement['distance_mm']:.1f}mm), "
              f"Action(R1:{action['rotate1_deg']:.1f}°, R2:{action['rotate2_deg']:.1f}°, D:{action['drive_mm']:.1f}mm)")
        
        # Check for collision
        if measurement['distance_mm'] < 300.0:
            print(f"    ⚠ Close to obstacle! Distance: {measurement['distance_mm']:.1f}mm")
            break
    
    print("✓ Simple policy test completed")

def test_policy_components():
    """Test individual policy components."""
    print("\nTesting Policy Components...")
    
    # Test policy network
    policy = SimplePolicy()
    
    # Test forward pass
    test_input = torch.randn(5)
    output = policy(test_input)
    
    print("✓ Policy forward pass works")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output: rotate1={output['rotate1_deg']:.2f}°, rotate2={output['rotate2_deg']:.2f}°, drive={output['drive_mm']:.2f}mm")
    
    # Test parameter access
    genome = []
    for param in policy.parameters():
        genome.extend(param.detach().numpy().flatten().tolist())
    
    print(f"✓ Policy parameter access works ({len(genome)} parameters)")

def main():
    """Main test function."""
    print("Policy Learning Component Test")
    print("=" * 40)
    
    # Test components
    test_policy_components()
    test_environment_simulator()
    test_simple_policy()
    
    print("\n" + "=" * 40)
    print("All tests completed!")

if __name__ == "__main__":
    main()