from Library import DataProcessor
import numpy as np
sessions = ['session03', 'session04', 'session06']

dc = DataProcessor.DataCollection(sessions)
dc.load_views(radius_mm=4000)
dc.load_profiles(az_steps=19)

views = dc.views
profiles = dc.profiles

# Data validation and sanity checks
print("\n=== DATA VALIDATION ===")
print(f"Views shape: {views.shape}")
print(f"Profiles shape: {profiles.shape}")
print(f"Number of samples: {len(views)}")

# Check for NaN values
views_nan_count = np.isnan(views).sum()
profiles_nan_count = np.isnan(profiles).sum()
print(f"NaN values in views: {views_nan_count}")
print(f"NaN values in profiles: {profiles_nan_count}")

# Check data ranges
print(f"Views min/max: {views.min()}/{views.max()}")
print(f"Profiles min/max: {profiles.min()}/{profiles.max()}")

# Check for zero/empty data
views_zero_count = (views == 0).sum()
profiles_zero_count = (profiles == 0).sum()
print(f"Zero values in views: {views_zero_count} ({views_zero_count / views.size * 100:.2f}%)")
print(f"Zero values in profiles: {profiles_zero_count} ({profiles_zero_count / profiles.size * 100:.2f}%)")

# Check for reasonable profile values (should be positive distances)
negative_profiles = (profiles < 0).sum()
print(f"Negative values in profiles: {negative_profiles}")

# Check for extremely large values
large_profiles = (profiles > 10000).sum()  # 10 meters seems reasonable max
print(f"Very large values (>10m) in profiles: {large_profiles}")

# Check view dimensions consistency
if views.ndim == 4 and views.shape[3] == 3:
    print("✓ Views have correct RGB format (N, H, W, 3)")
else:
    print("✗ Views do not have expected RGB format")

# Check profiles are 2D
if profiles.ndim == 2:
    print("✓ Profiles have correct 2D format (N, az_steps)")
else:
    print("✗ Profiles do not have expected 2D format")

# Sample some specific indices to verify data quality
print("\n=== SAMPLE DATA CHECK ===")
sample_indices = [0, 50, 100, -1]  # First, middle, and last samples
for idx in sample_indices:
    print(f"Sample {idx}:")
    print(f"  View shape: {views[idx].shape}, dtype: {views[idx].dtype}")
    print(f"  View min/max: {views[idx].min()}/{views[idx].max()}")
    print(f"  Profile shape: {profiles[idx].shape}, dtype: {profiles[idx].dtype}")
    print(f"  Profile min/max: {profiles[idx].min()}/{profiles[idx].max()}")
    print(f"  Profile mean: {profiles[idx].mean():.2f}")

# Check for consistent data across sessions
print(f"\n=== SESSION CONSISTENCY CHECK ===")
session_sizes = []
for i, session in enumerate(sessions):
    session_views = dc.processors[i].views
    session_profiles = dc.processors[i].profiles
    session_sizes.append(len(session_views))
    print(f"Session {session}: {len(session_views)} views, {len(session_profiles)} profiles")

print(f"Total samples from sessions: {sum(session_sizes)}")
print(f"Total samples in collection: {len(views)}")
if sum(session_sizes) == len(views):
    print("✓ Sample counts match across sessions")
else:
    print("✗ Sample count mismatch!")

# Bounding box calculation and cropping
print(f"\n=== BOUNDING BOX ANALYSIS ===")
bbox = DataProcessor.find_bounding_box_across_views(views)
if bbox:
    x_min, y_min, x_max, y_max = bbox
    print(f"Bounding box: x({x_min}:{x_max}), y({y_min}:{y_max})")
    print(f"Bounding box size: width={x_max-x_min}, height={y_max-y_min}")
    
    cropped = views[:, y_min:y_max, x_min:x_max, :]
    print(f"Cropped views shape: {cropped.shape}")
    print(f"Memory reduction: {(1 - cropped.size/views.size)*100:.1f}%")
else:
    print("No bounding box found - all pixels might be black")

# Calculate closest distance from profiles
closest_distance = np.min(profiles, axis=1)
print(f"\n=== CLOSEST DISTANCE ANALYSIS ===")
print(f"Closest distance shape: {closest_distance.shape}")
print(f"Closest distance min/max: {closest_distance.min():.2f}/{closest_distance.max():.2f}")
print(f"Closest distance mean: {closest_distance.mean():.2f}")
print(f"Closest distance std: {closest_distance.std():.2f}")

# Check for potential issues
issues_found = []
if views_nan_count > 0:
    issues_found.append(f"Found {views_nan_count} NaN values in views")
if profiles_nan_count > 0:
    issues_found.append(f"Found {profiles_nan_count} NaN values in profiles")
if negative_profiles > 0:
    issues_found.append(f"Found {negative_profiles} negative values in profiles")
if large_profiles > 0:
    issues_found.append(f"Found {large_profiles} very large (>10m) values in profiles")

if issues_found:
    print(f"\n⚠️  POTENTIAL ISSUES FOUND:")
    for issue in issues_found:
        print(f"  - {issue}")
else:
    print(f"\n✅ NO OBVIOUS ISSUES FOUND - DATA LOOKS GOOD!")

print(f"\n=== DATA SUMMARY ===")
print(f"Successfully loaded {len(views)} samples")
print(f"Views: {views.shape} ({views.nbytes / 1024 / 1024:.1f} MB)")
print(f"Profiles: {profiles.shape} ({profiles.nbytes / 1024 / 1024:.1f} MB)")
print(f"Total memory usage: {(views.nbytes + profiles.nbytes) / 1024 / 1024:.1f} MB")