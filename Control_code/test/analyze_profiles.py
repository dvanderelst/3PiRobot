#!/usr/bin/env python3

"""
Analyze profile values to understand the clipping issue.
"""

import numpy as np
import sys
sys.path.append('/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code')

from Library import DataProcessor

def analyze_profiles():
    """Analyze profile values."""
    print("Analyzing profile values...")
    
    dc = DataProcessor.DataCollection(['session07'])
    eval_proc = None
    
    for p in dc.processors:
        if 'session07' in p.session:
            eval_proc = p
            break
    
    if eval_proc is None:
        print("❌ Could not find session07 processor")
        return
    
    # Load profiles
    eval_proc.load_profiles(opening_angle=45, steps=9)
    
    print(f"Loaded {len(eval_proc.profiles)} profiles")
    
    # Analyze first 10 profiles
    print("\nFirst 10 profiles:")
    for i in range(10):
        profile = eval_proc.profiles[i]
        print(f"Profile {i}: min={profile.min():.1f}, max={profile.max():.1f}, mean={profile.mean():.1f}")
        
        # Check clipping at different thresholds
        clipped_1450 = np.where(profile > 1450.0, np.nan, profile)
        clipped_3000 = np.where(profile > 3000.0, np.nan, profile)
        
        print(f"  Clipped at 1450: {np.sum(~np.isnan(clipped_1450))}/9 valid points")
        print(f"  Clipped at 3000: {np.sum(~np.isnan(clipped_3000))}/9 valid points")
    
    # Overall statistics
    all_profiles = eval_proc.profiles.flatten()
    print(f"\nOverall statistics:")
    print(f"All profiles: min={all_profiles.min():.1f}, max={all_profiles.max():.1f}, mean={all_profiles.mean():.1f}")
    print(f"Median: {np.median(all_profiles):.1f}")
    print(f"25th percentile: {np.percentile(all_profiles, 25):.1f}")
    print(f"75th percentile: {np.percentile(all_profiles, 75):.1f}")
    print(f"95th percentile: {np.percentile(all_profiles, 95):.1f}")
    
    # Check what percentage would be clipped at different thresholds
    thresholds = [1450, 1500, 2000, 3000]
    for threshold in thresholds:
        clipped_count = np.sum(all_profiles > threshold)
        percentage = (clipped_count / len(all_profiles)) * 100
        print(f"Clipped at {threshold}: {clipped_count}/{len(all_profiles)} ({percentage:.1f}%)")

if __name__ == "__main__":
    analyze_profiles()