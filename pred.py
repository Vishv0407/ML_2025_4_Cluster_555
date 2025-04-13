import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import os

# Constants
INTERSECTION_DISTANCE_THRESHOLD = 50  # meters
VELOCITY_THRESHOLD = 0.1  # m/s (adjustable)
FRAME_RATE = 30  # frames per second (adjustable based on data)

# Load and process data
def load_trajectories(file_paths):
    data = {}
    for file in file_paths:
        try:
            df = pd.read_csv(file)
            trajectory_id = os.path.basename(file).replace('.csv', '')
            data[trajectory_id] = df[['x', 'y']].values  # Assuming x, y coordinates
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return data

# Calculate average velocity (m/s)
def calculate_velocity(trajectory):
    velocities = []
    for i in range(len(trajectory) - 1):
        dist = distance.euclidean(trajectory[i], trajectory[i + 1])
        time_diff = 1 / FRAME_RATE  # Time between frames
        velocity = dist / time_diff
        velocities.append(velocity)
    return np.mean(velocities) if velocities else 0

# Check if parked near intersection
def is_parked_near_intersection(trajectory, intersection_points):
    start_point = trajectory[0]
    for point in intersection_points:
        if distance.euclidean(start_point, point) < INTERSECTION_DISTANCE_THRESHOLD:
            return True
    return False

# Rule-based classification
def classify_trajectory(trajectory, reason, zone_id):
    # Intersection points (placeholder, adjust based on actual data)
    intersection_points = [(0, 0), (100, 0), (0, 100), (100, 100)]
    
    velocity = calculate_velocity(trajectory)
    is_stationary = velocity < VELOCITY_THRESHOLD
    
    if 'wrong side entry' in reason.lower():
        return 'abnormal', 'Wrong side entry'
    elif 'anti-clockwise' in reason.lower():
        return 'abnormal', 'Anti-clockwise traversal'
    elif 'parking' in reason.lower() and is_stationary and is_parked_near_intersection(trajectory, intersection_points):
        return 'abnormal', 'Parking near intersection'
    else:
        return 'normal', 'Valid trajectory'

# DBSCAN clustering for valid trajectories
def cluster_valid_trajectories(trajectories):
    X = np.array([traj.mean(axis=0) for traj in trajectories])  # Use mean x, y as features
    db = DBSCAN(eps=10, min_samples=5).fit(X)  # Adjust eps and min_samples
    labels = db.labels_
    return labels

# Main processing
data_dir = "Codes/dataset_interpolated/processed/interpolated"
input_data = []

# Walk through the directory structure
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            zone_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            true_label = 'normal' if 'normal' in os.path.basename(os.path.dirname(file_path)) else 'abnormal'
            # Placeholder for reason; update if available
            reason = 'wrong side entry' if 'abnormal' in true_label else ''
            input_data.append(f"{file_path}, True: {true_label}, Pred: , Reason: {reason}, Zone: {zone_id}")

file_data = [
    {'file': f, 'true': t, 'pred': p, 'reason': r, 'zone': int(z)}
    for f, t, p, r, z in [
        (line.split(', True: ')[0], line.split(', True: ')[1].split(', Pred: ')[0],
         line.split(', Pred: ')[1].split(', Reason: ')[0], line.split(', Reason: ')[1].split(', Zone: ')[0],
         line.split(', Zone: ')[1])
        for line in input_data if line.strip()
    ] if f
]

trajectories = load_trajectories([d['file'] for d in file_data])
results = []

for data in file_data:
    trajectory_id = os.path.basename(data['file']).replace('.csv', '')
    if trajectory_id in trajectories:
        trajectory = trajectories[trajectory_id]
        pred, reason = classify_trajectory(trajectory, data['reason'], data['zone'])
        results.append({
            'file': data['file'],
            'true': data['true'],
            'pred': pred,
            'reason': reason,
            'zone': data['zone']
        })

# Clustering valid trajectories
valid_trajectories = [trajectories[os.path.basename(d['file']).replace('.csv', '')] 
                      for d in results if d['pred'] == 'normal']
if valid_trajectories:
    clusters = cluster_valid_trajectories(valid_trajectories)
    print("Cluster labels for valid trajectories:", clusters)

# Output results
for result in results:
    print(f"File: {result['file']}, True: {result['true']}, Pred: {result['pred']}, Reason: {result['reason']}")