import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import psutil  # For memory monitoring
import gc  # For garbage collection
from sklearn.cluster import DBSCAN

np.random.seed(42)

# Normalize coordinates using UAV geospatial data
def normalize_coordinates(df: pd.DataFrame, long: float, lat: float, alt: float, global_width: float = 1920, global_height: float = 1080) -> pd.DataFrame:
    alt_scale = alt / 100.0
    df['x_global'] = (df['left'] + df['w'] / 2) * alt_scale
    df['y_global'] = (df['top'] + df['h'] / 2) * alt_scale
    df['x_local'] = df['x_global'] - np.mean(df['x_global'])
    df['y_local'] = df['y_global'] - np.mean(df['y_global'])
    return df

# Convert DataFrame to points in local coordinates
def df_to_points(df: pd.DataFrame) -> List[Tuple[int, float, float]]:
    return [(int(row['frameNo']), row['x_local'], row['y_local']) for _, row in df.iterrows()]

# Check if trajectory intersects the roundabout
def is_roundabout_trajectory(points: List[Tuple[int, float, float]], center_x: float, center_y: float, radius: float) -> bool:
    distances = [np.sqrt((p[1] - center_x)**2 + (p[2] - center_y)**2) for p in points]
    return any(d < radius * 1.5 for d in distances)

# Derive roundabout geometry using a sample of turning trajectories
def derive_roundabout_geometry(normal_trajectories: List[pd.DataFrame], max_samples: int = 100) -> Tuple[float, float, float]:
    turning_trajectories = []
    for traj_df in normal_trajectories[:min(len(normal_trajectories), max_samples)]:
        points = df_to_points(traj_df)
        if len(points) < 2:
            continue
        angles = []
        center_x_temp, center_y_temp = np.mean([(p[1], p[2]) for p in points], axis=0)
        for p in points:
            dx = p[1] - center_x_temp
            dy = p[2] - center_y_temp
            angle = np.degrees(np.arctan2(dy, dx)) % 360
            angles.append(angle)
        angle_change = max(angles) - min(angles) if max(angles) > min(angles) else 360 + max(angles) - min(angles)
        if angle_change > 90:
            turning_trajectories.append(traj_df)
    
    if not turning_trajectories:
        print("Warning: No normal trajectories with significant turning found. Using all normal trajectories (limited).")
        turning_trajectories = normal_trajectories[:max_samples]
    
    all_points = [p for traj_df in turning_trajectories for p in df_to_points(traj_df)]
    if not all_points:
        return 0.0, 0.0, 100.0
    points_array = np.array([(p[1], p[2]) for p in all_points])
    center_x, center_y = np.mean(points_array, axis=0)
    db = DBSCAN(eps=50, min_samples=10).fit(points_array)
    labels = db.labels_
    core_points = points_array[labels != -1]
    if len(core_points) == 0:
        distances = np.sqrt((points_array[:, 0] - center_x)**2 + (points_array[:, 1] - center_y)**2)
        radius = np.percentile(distances, 75)
    else:
        distances = np.sqrt((core_points[:, 0] - center_x)**2 + (core_points[:, 1] - center_y)**2)
        radius = np.median(distances)
    return center_x, center_y, radius * 0.9

# Assign zone and lane based on local coordinates
def assign_zone_and_lane(points: List[Tuple[int, float, float]], center_x: float, center_y: float, radius: float) -> List[Tuple[int, int, str, str, str]]:
    lane_width = radius / 3
    zone_assignments = []
    for frame, x, y in points:
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        angle = np.degrees(np.arctan2(y - center_y, x - center_x)) % 360
        
        if distance < 0.3 * radius:
            zone, direction, lane, part = 0, 'none', 'none', 'central'
        elif distance < radius:
            zone = 1
            direction = 'circulating'
            lane = 'innermost' if distance < lane_width else 'middle' if distance < 2 * lane_width else 'outermost'
            part = 'circulating'
        else:
            zone = {0: 2, 90: 3, 180: 4, 270: 5}[int(angle // 90) * 90]
            part = ['North', 'East', 'South', 'West'][int(angle // 90)]
            direction = 'inbound' if distance < 2.0 * radius else 'outbound'
            lane = 'innermost' if distance < radius + lane_width else 'middle' if distance < radius + 2 * lane_width else 'outermost'
        
        zone_assignments.append((frame, zone, direction, lane, part))
    return zone_assignments

# Get zone sequence
def get_zone_sequence(points: List[Tuple[int, float, float]], center_x: float, center_y: float, radius: float) -> List[int]:
    return [z[1] for z in assign_zone_and_lane(points, center_x, center_y, radius)]

# Define valid trajectories based on lane rules
def is_valid_trajectory(zone_assignments: List[Tuple[int, int, str, str, str]]) -> Tuple[bool, str]:
    direction_sequence = [z[2] for z in zone_assignments]
    lane_sequence = [z[3] for z in zone_assignments]
    part_sequence = [z[4] for z in zone_assignments]
    
    start_lane = lane_sequence[0] if lane_sequence else None
    start_part = part_sequence[0] if part_sequence else None
    end_lane = lane_sequence[-1] if lane_sequence else None
    end_part = part_sequence[-1] if part_sequence else None
    
    # Handle circulating lane case
    if start_part == 'circulating' or end_part == 'circulating':
        return True, 'Valid trajectory (circulating lane)'
    
    part_relations = {
        'North': {'A': ['North'], 'R': ['East'], 'O': ['South'], 'L': ['West']},
        'East': {'A': ['East'], 'R': ['South'], 'O': ['West'], 'L': ['North']},
        'South': {'A': ['South'], 'R': ['West'], 'O': ['North'], 'L': ['East']},
        'West': {'A': ['West'], 'R': ['North'], 'O': ['East'], 'L': ['South']}
    }
    
    if start_lane and start_part and end_lane and end_part:
        if start_lane == 'outermost':  # Incoming from iii
            if not (end_part in part_relations[start_part]['L'] and end_lane == 'outermost'):
                return False, 'Invalid left turn from outermost lane'
        elif start_lane == 'middle':  # Incoming from ii
            if not (end_part in part_relations[start_part]['O'] and end_lane == 'middle'):
                return False, 'Invalid straight path from middle lane'
        elif start_lane == 'innermost':  # Incoming from i
            valid_ends = (end_part in part_relations[start_part]['R'] and end_lane == 'innermost') or \
                         (end_part in part_relations[start_part]['O'] and end_lane == 'innermost') or \
                         (end_part in part_relations[start_part]['A'] and end_lane == 'innermost')
            if not valid_ends:
                return False, 'Invalid path from innermost lane'
    
    return True, 'Valid trajectory'

# Rule-based classification
def classify_trajectory(trajectory: pd.DataFrame, center_x: float, center_y: float, radius: float) -> Tuple[str, str]:
    points = df_to_points(trajectory)
    zone_assignments = assign_zone_and_lane(points, center_x, center_y, radius)
    direction_sequence = [z[2] for z in zone_assignments]
    lane_sequence = [z[3] for z in zone_assignments]
    
    entry_index = min(10, len(direction_sequence) // 3)
    for i, (direction, lane) in enumerate(zip(direction_sequence, lane_sequence)):
        if lane == 'none':
            continue
        if i < entry_index:
            if direction == 'outbound' and lane in ['innermost', 'middle', 'outermost']:
                return 'abnormal', 'incoming from outgoing lane'
            elif direction == 'inbound' and lane in ['innermost', 'middle', 'outermost']:
                return 'abnormal', 'incoming from incoming lane unexpectedly'

    if len(points) > 10:
        angles = [np.degrees(np.arctan2(p[2] - center_y, p[1] - center_x)) % 360 for p in points]
        angle_diffs = [((angles[i+1] - angles[i]) % 360) for i in range(len(angles)-1)]
        if len(angle_diffs) > 0 and sum(d < 180 for d in angle_diffs) / len(angle_diffs) > 0.7:
            return 'abnormal', 'anti-clockwise traversal'

    intersection_buffer = 50.0
    for i in range(1, len(points)):
        dx = points[i][1] - points[i-1][1]
        dy = points[i][2] - points[i-1][2]
        interframe_distance = np.sqrt(dx**2 + dy**2)
        mean_distance = np.mean([np.sqrt((points[j][1] - points[j-1][1])**2 + (points[j][2] - points[j-1][2])**2) 
                               for j in range(1, len(points)) if j > i-5 and j < i+5]) if i > 0 and i < len(points)-1 else interframe_distance
        distance_to_center = np.sqrt((points[i][1] - center_x)**2 + (points[i][2] - center_y)**2)
        if distance_to_center > radius and distance_to_center < radius + intersection_buffer and mean_distance < 0.1:
            return 'abnormal', 'parking near intersection'

    is_valid, reason = is_valid_trajectory(zone_assignments)
    if not is_valid:
        return 'abnormal', reason

    return 'normal', 'valid trajectory'

# Extract features with batch processing
def extract_features(trajectories: List[pd.DataFrame], labels: List[int], file_paths: List[str], fps: float = 1.0, batch_size: int = 100) -> pd.DataFrame:
    normal_trajectories = [traj for traj, label in zip(trajectories, labels) if label == 0][:100]
    center_x, center_y, radius = derive_roundabout_geometry(normal_trajectories)
    results = []
    
    for i in range(0, len(trajectories), batch_size):
        batch_trajectories = trajectories[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        batch_file_paths = file_paths[i:i + batch_size]
        
        for traj_df, label, file_path in zip(batch_trajectories, batch_labels, batch_file_paths):
            classification, reason = classify_trajectory(traj_df, center_x, center_y, radius)
            pred_label = 0 if classification == 'normal' else 1
            results.append({
                'track_id': len(results) + 1,
                'true_label': label,
                'pred_label': pred_label,
                'file_path': file_path,
                'classification_reason': reason
            })
        gc.collect()
        memory_percent = psutil.virtual_memory().percent
        print(f"Processed batch {i//batch_size + 1}, Memory usage: {memory_percent}%")
        if memory_percent > 80:
            print("Warning: Memory usage exceeds 80%. Consider reducing batch_size or data.")
    
    return pd.DataFrame(results)

# Clean DataFrame
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# Load trajectories with geospatial filtering and batching
def load_trajectories(video_folder: str, min_points: int = 7, batch_size: int = 100) -> Tuple[List[pd.DataFrame], List[int], List[str]]:
    temp_trajectories, temp_labels, temp_file_paths = [], [], []
    for root, dirs, _ in os.walk(video_folder):
        if 'normal' in dirs and 'abnormal' in dirs:
            numbered_dir = os.path.basename(root)
            print(f"Processing directory: {numbered_dir}")
            for folder, label in [('normal', 0), ('abnormal', 1)]:
                path = os.path.join(root, folder)
                files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
                print(f"Found {len(files)} {folder} CSV files")
                for file in files:
                    df = pd.read_csv(file)[['frameNo', 'left', 'top', 'w', 'h', 'long', 'lat', 'alt']].sort_values(by='frameNo')
                    if len(df) >= min_points:
                        long = np.mean(df['long'])
                        lat = np.mean(df['lat'])
                        alt = np.mean(df['alt'])
                        df_normalized = normalize_coordinates(df, long, lat, alt)
                        temp_trajectories.append(df_normalized)
                        temp_labels.append(label)
                        temp_file_paths.append(file)
                    if len(temp_trajectories) % batch_size == 0 and len(temp_trajectories) > 0:
                        gc.collect()
                        print(f"Loaded {len(temp_trajectories)} trajectories so far, Memory usage: {psutil.virtual_memory().percent}%")

    normal_trajectories = [traj for traj, label in zip(temp_trajectories, temp_labels) if label == 0]
    center_x, center_y, radius = derive_roundabout_geometry(normal_trajectories)

    trajectories, labels, file_paths = [], [], []
    for traj_df, label, file in zip(temp_trajectories, temp_labels, temp_file_paths):
        points = df_to_points(traj_df)
        if is_roundabout_trajectory(points, center_x, center_y, radius):
            trajectories.append(traj_df)
            labels.append(label)
            file_paths.append(file)
    return trajectories, labels, file_paths

# Plot individual trajectory
def plot_individual_trajectory(file_path: str, center_x: float, center_y: float, radius: float, title: str):
    df = pd.read_csv(file_path)
    long = np.mean(df['long'])
    lat = np.mean(df['lat'])
    alt = np.mean(df['alt'])
    df = normalize_coordinates(df, long, lat, alt)
    points = df_to_points(df)
    plt.figure(figsize=(8, 8))
    circle = plt.Circle((center_x, center_y), radius, color='g', fill=False, label='Circulating Lane')
    island = plt.Circle((center_x, center_y), radius * 0.25, color='r', fill=True, label='Central Island')
    plt.gca().add_patch(circle)
    plt.gca().add_patch(island)
    x, y = zip(*[(p[1], p[2]) for p in points])
    plt.plot(x, y, 'b-', label='Trajectory')
    plt.title(title)
    plt.legend()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    video_folder = "Codes/dataset_interpolated/processed/interpolated"
    print(f"Using video_folder: {video_folder}")

    try:
        trajectories, labels, file_paths = load_trajectories(video_folder, min_points=7, batch_size=100)
        print(f"Loaded {len(trajectories)} trajectories after filtering.")

        if len(trajectories) == 0:
            print("No trajectories loaded. Check directory structure.")
        else:
            fps = 1.0
            features_df = extract_features(trajectories, labels, file_paths, fps, batch_size=100)
            print("Feature extraction complete. Sample data:")
            print(features_df.head())

            print("\nClassification Results:")
            print(features_df[['file_path', 'true_label', 'pred_label', 'classification_reason']])

            normal_trajectories = [traj for traj, label in zip(trajectories, labels) if label == 0]
            center_x, center_y, radius = derive_roundabout_geometry(normal_trajectories)
            print(f"Derived Roundabout Geometry - Center: ({center_x}, {center_y}), Radius: {radius}")

            print("\nPlotting Sample Trajectories:")
            if len(trajectories) > 0:
                sample_file = file_paths[0]
                print(f"Sample Trajectory: {sample_file}")
                plot_individual_trajectory(sample_file, center_x, center_y, radius, "Sample Trajectory")
    except MemoryError:
        print("MemoryError: Process exceeded available memory. Try reducing batch_size or min_points.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        gc.collect()
        print(f"Final memory usage: {psutil.virtual_memory().percent}%")