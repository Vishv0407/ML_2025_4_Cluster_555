import numpy as np
import pandas as pd
import os
from typing import List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib

# Constants
FPS = 30.0
PIXEL_TO_METER = 0.05
LANE_WIDTH = 3.5
INTERSECTION_RADIUS = 50.0
OUTPUT_DIR = "results"
MODEL_DIR = "model_1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(root_dir: str) -> pd.DataFrame:
    dataframes = []
    for folder in ['10', '11', '12']:
        for label, subfolder in [(0, 'normal'), (1, 'abnormal')]:
            path = os.path.join(root_dir, folder, subfolder)
            if not os.path.exists(path):
                print(f"Warning: {path} does not exist")
                continue
            for file in os.listdir(path):
                if file.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(path, file))
                        required_cols = ['frameNo', 'left', 'top', 'w', 'h']
                        if not all(col in df.columns for col in required_cols):
                            print(f"Skipping {file}: missing required columns")
                            continue
                        df['vehicle_id'] = f"{folder}_{subfolder}_{file[:-4]}"
                        df['label'] = label
                        dataframes.append(df)
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
    if not dataframes:
        raise ValueError("No valid CSV files found")
    return pd.concat(dataframes, ignore_index=True)

def derive_roundabout_geometry(df: pd.DataFrame) -> Tuple[float, float, float]:
    points = (df[['left', 'top']].values + df[['w', 'h']].values / 2) * PIXEL_TO_METER
    if len(points) < 3:
        raise ValueError("Insufficient points for geometry")
    center_x, center_y = np.mean(points, axis=0)
    radius = np.mean(np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2))
    return center_x, center_y, max(radius, 20.0)

def identify_lane_and_direction(points: List[Tuple[int, float, float]], center_x: float, center_y: float, radius: float) -> Tuple[str, str, str, str]:
    lane_width = radius / 3.5
    for point, prefix in [(points[0], 'start'), (points[-1], 'end')]:
        distance = np.sqrt((point[1] - center_x)**2 + (point[2] - center_y)**2)
        if radius * 0.15 <= distance < radius * 0.15 + lane_width:
            lane = 'i'
        elif radius * 0.15 + lane_width <= distance < radius * 0.15 + 2*lane_width:
            lane = 'ii'
        elif radius * 0.15 + 2*lane_width <= distance < radius * 0.15 + 3*lane_width:
            lane = 'iii'
        else:
            lane = 'unknown'
        angle = np.degrees(np.arctan2(point[2] - center_y, point[1] - center_x)) % 360
        if 45 <= angle < 135:
            direction = 'East'
        elif 135 <= angle < 225:
            direction = 'South'
        elif 225 <= angle < 315:
            direction = 'West'
        else:
            direction = 'North'
        if prefix == 'start':
            start_lane, start_direction = lane, direction
        else:
            end_lane, end_direction = lane, direction
    return start_lane, start_direction, end_lane, end_direction

def get_zone_sequence(points: List[Tuple[int, float, float]], center_x: float, center_y: float, radius: float) -> List[int]:
    zone_seq = []
    lane_width = radius / 3.5
    for point in points:
        distance = np.sqrt((point[1] - center_x)**2 + (point[2] - center_y)**2)
        angle = np.degrees(np.arctan2(point[2] - center_y, point[1] - center_x)) % 360
        if distance < radius * 0.15:
            zone_seq.append(0)
        elif radius * 0.15 <= distance < radius * 0.15 + 3*lane_width:
            zone_seq.append(1)
        else:
            if 45 <= angle < 135:
                zone_seq.append(3)
            elif 135 <= angle < 225:
                zone_seq.append(4)
            elif 225 <= angle < 315:
                zone_seq.append(5)
            else:
                zone_seq.append(2)
    return zone_seq

def extract_features(points: List[Tuple[int, float, float]], center_x: float, center_y: float, radius: float, fps: float = FPS) -> dict:
    start_lane, start_direction, end_lane, end_direction = identify_lane_and_direction(points, center_x, center_y, radius)
    trajectory_type = f"{start_direction}_{start_lane}_to_{end_direction}_{end_lane}"
    zone_seq = get_zone_sequence(points, center_x, center_y, radius)
    
    distances = [np.sqrt((p[1] - center_x)**2 + (p[2] - center_y)**2) for p in points]
    curvature_adherence = np.var(distances) / (radius**2) if distances else 0.0
    lane_changes = sum(1 for i in range(1, len(zone_seq)) if zone_seq[i] == 1 and zone_seq[i-1] != 1)
    forbidden_transitions = 1 if 0 in zone_seq else 0
    
    path_length = sum(np.sqrt((points[i][1] - points[i-1][1])**2 + (points[i][2] - points[i-1][2])**2) 
                      for i in range(1, len(points)))
    duration = (points[-1][0] - points[0][0]) / fps if len(points) > 1 else 1.0
    path_length = path_length / duration if duration > 0 else 0.0
    
    angles = []
    for i in range(1, len(points)):
        x1, y1 = points[i-1][1] - center_x, points[i-1][2] - center_y
        x2, y2 = points[i][1] - center_x, points[i][2] - center_y
        angle1 = np.arctan2(y1, x1)
        angle2 = np.arctan2(y2, x2)
        angle_diff = (angle2 - angle1) % (2 * np.pi)
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        angles.append(angle_diff)
    angle_variance = np.var(angles) if angles else 0.0
    
    speeds = []
    accelerations = []
    for i in range(1, len(points)):
        dx = points[i][1] - points[i-1][1]
        dy = points[i][2] - points[i-1][2]
        distance = np.sqrt(dx**2 + dy**2)
        time = (points[i][0] - points[i-1][0]) / fps
        speed = distance / time if time > 0 else 0.0
        speeds.append(speed)
        if i > 1:
            accel = (speeds[-1] - speeds[-2]) / time if time > 0 else 0.0
            accelerations.append(accel)
    mean_speed = np.mean(speeds) if speeds else 0.0
    speed_variance = np.var(speeds) if speeds else 0.0
    accel_variance = np.var(accelerations) if accelerations else 0.0
    
    interframe_distances = [np.sqrt((points[i][1] - points[i-1][1])**2 + (points[i][2] - points[i-1][2])**2) 
                           for i in range(1, len(points))]
    mean_distance = np.mean(interframe_distances) if interframe_distances else 0.0
    variance_distance = np.var(interframe_distances) if interframe_distances else 0.0
    total_displacement = np.sqrt((points[-1][1] - points[0][1])**2 + (points[-1][2] - points[0][2])**2)
    is_stationary = 1 if mean_distance < 0.3 and variance_distance < 0.05 and total_displacement < 0.8 else 0
    
    anti_clockwise = 0
    if any(z == 1 for z in zone_seq) and sum(angles) > 0:
        anti_clockwise = 1
    
    corner_proximity = 0
    corners = [
        (center_x + INTERSECTION_RADIUS, center_y + INTERSECTION_RADIUS),
        (center_x + INTERSECTION_RADIUS, center_y - INTERSECTION_RADIUS),
        (center_x - INTERSECTION_RADIUS, center_y - INTERSECTION_RADIUS),
        (center_x - INTERSECTION_RADIUS, center_y + INTERSECTION_RADIUS)
    ]
    if is_stationary:
        for point in points:
            for cx, cy in corners:
                dist = np.sqrt((point[1] - cx)**2 + (point[2] - cy)**2)
                if dist < 50.0:
                    corner_proximity = 1
                    break
    
    return {
        'trajectory_type': trajectory_type,
        'curvature_adherence': curvature_adherence,
        'lane_changes': lane_changes,
        'forbidden_transitions': forbidden_transitions,
        'mean_speed': mean_speed,
        'speed_variance': speed_variance,
        'accel_variance': accel_variance,
        'is_stationary': is_stationary,
        'anti_clockwise': anti_clockwise,
        'corner_proximity': corner_proximity,
        'path_length': path_length,
        'angle_variance': angle_variance
    }

def prepare_features(df: pd.DataFrame, center_x: float, center_y: float, radius: float) -> pd.DataFrame:
    features_list = []
    for vehicle_id, group in df.groupby('vehicle_id'):
        try:
            points = [(row['frameNo'], (row['left'] + row['w']/2) * PIXEL_TO_METER, 
                       (row['top'] + row['h']/2) * PIXEL_TO_METER) for _, row in group.iterrows()]
            if len(points) < 50:
                print(f"Skipping vehicle {vehicle_id}: too few points ({len(points)})")
                continue
            print(f"Vehicle {vehicle_id}: {len(points)} points processed")
            features = extract_features(points, center_x, center_y, radius)
            features['vehicle_id'] = vehicle_id
            features['label'] = group['label'].iloc[0]
            features_list.append(features)
        except Exception as e:
            print(f"Error processing vehicle {vehicle_id}: {e}")
    features_df = pd.DataFrame(features_list)
    
    if features_df.empty:
        print("No valid trajectories after feature extraction")
        return features_df
    
    return features_df

def plot_trajectories(df: pd.DataFrame, center_x: float, center_y: float, radius: float, labels: np.ndarray, features_df: pd.DataFrame):
    plt.figure(figsize=(10, 10))
    valid_vehicles = set(features_df['vehicle_id'])
    for vehicle_id, group in df.groupby('vehicle_id'):
        if vehicle_id not in valid_vehicles:
            continue
        points = [((row['left'] + row['w']/2) * PIXEL_TO_METER, 
                   (row['top'] + row['h']/2) * PIXEL_TO_METER) for _, row in group.iterrows()]
        x, y = zip(*points)
        vehicle_idx = features_df[features_df['vehicle_id'] == vehicle_id].index
        if vehicle_idx.empty:
            continue
        idx = vehicle_idx[0]
        color = 'r' if labels[idx] == 1 else 'g'
        plt.plot(x, y, color, alpha=0.5)
    
    circle = plt.Circle((center_x, center_y), radius, fill=False, color='b')
    plt.gca().add_patch(circle)
    plt.gca().set_aspect('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectories: Green=Normal, Red=Abnormal')
    plt.savefig(os.path.join(OUTPUT_DIR, 'trajectories.png'))
    plt.close()

def main(root_dir: str):
    try:
        df = load_data(root_dir)
        print(f"Loaded {len(df['vehicle_id'].unique())} trajectories")
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    
    try:
        center_x, center_y, radius = derive_roundabout_geometry(df)
        print(f"Center=({center_x:.2f}, {center_y:.2f}), radius={radius:.2f}m")
    except ValueError as e:
        print(f"Error in geometry: {e}")
        return
    
    features_df = prepare_features(df, center_x, center_y, radius)
    if features_df.empty:
        print("No valid trajectories after feature extraction")
        return
    
    numeric_features = [
        'curvature_adherence', 'lane_changes', 'forbidden_transitions',
        'mean_speed', 'speed_variance', 'accel_variance',
        'is_stationary', 'anti_clockwise', 'corner_proximity',
        'path_length', 'angle_variance'
    ]
    
    X = features_df[numeric_features].values
    y = features_df['label'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_scaled, y = smote.fit_resample(X_scaled, y)
    print(f"After SMOTE: {len(y)} samples, {sum(y==0)} normal, {sum(y==1)} abnormal")
    
    rf_clf = RandomForestClassifier(
        random_state=42,
        n_estimators=400,
        max_depth=20,
        class_weight={0:1, 1:2}
    )
    
    # Use 10-fold to get folds 3-7
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_reports = []
    train_indices = []
    test_indices = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        if fold in [2, 3, 4, 5, 6]:  # Folds 3-7 (0-based: 2-6)
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            rf_clf.fit(X_train, y_train)
            y_pred = rf_clf.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            fold_accuracies.append(accuracy)
            fold_reports.append(classification_report(y_test, y_pred, output_dict=True))
            train_indices.extend(np.array(train_idx, dtype=int).tolist())
            test_indices.extend(np.array(test_idx, dtype=int).tolist())
            print(f"\nFold {fold+1} Accuracy: {accuracy:.4f}")
            print(f"Fold {fold+1} Classification Report:")
            print(classification_report(y_test, y_pred))
    
    # Compute average accuracy
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"\nAverage Accuracy (Folds 3-7): {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    
    # Train final model on combined folds 3-7
    train_indices = np.unique(np.array(train_indices, dtype=int))
    test_indices = np.unique(np.array([i for i in test_indices if i not in train_indices], dtype=int))
    
    if len(test_indices) == 0:
        print("Warning: No unique test indices after overlap removal. Using last fold's test set.")
        _, test_indices = list(skf.split(X_scaled, y))[6]  # Use Fold 7's test set as fallback
    
    X_train, y_train = X_scaled[train_indices], y[train_indices]
    X_test, y_test = X_scaled[test_indices], y[test_indices]
    
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    final_accuracy = (y_pred == y_test).mean()
    final_report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\nFinal Model Accuracy (Folds 3-7 Combined): {final_accuracy:.4f}")
    print(f"Final Model Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model, scaler, and geometry
    joblib.dump(rf_clf, os.path.join(MODEL_DIR, 'rf_model_fold3to7.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump((center_x, center_y, radius), os.path.join(MODEL_DIR, 'geometry.pkl'))
    print(f"Model, scaler, and geometry saved to {MODEL_DIR}")
    
    # Evaluate on full data for consistency
    y_pred_full = rf_clf.predict(X_scaled)
    print(f"\nFull Data Confusion Matrix:")
    print(confusion_matrix(y, y_pred_full))
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"Average Accuracy (Folds 3-7): {avg_accuracy:.4f} (±{std_accuracy:.4f})\n")
        f.write("\nFinal Model Accuracy (Folds 3-7 Combined): {final_accuracy:.4f}\n")
        f.write("\nFinal Model Classification Report:\n")
        for cls in ['0', '1']:
            f.write(f"Class {cls}: Precision={final_report[cls]['precision']:.2f}, "
                    f"Recall={final_report[cls]['recall']:.2f}, F1={final_report[cls]['f1-score']:.2f}\n")
        f.write("\nFold Accuracies:\n")
        for i, acc in enumerate(fold_accuracies, 3):
            f.write(f"Fold {i}: {acc:.4f}\n")
        f.write("\nFull Data Confusion Matrix:\n")
        f.write(str(confusion_matrix(y, y_pred_full)))
    
    plot_trajectories(df, center_x, center_y, radius, y_pred_full, features_df)
    
    importances = pd.DataFrame({
        'feature': numeric_features,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nFeature Importances:")
    print(importances)
    importances.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    try:
        df = load_data(root_dir)
        print(f"Loaded {len(df['vehicle_id'].unique())} trajectories")
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    
    try:
        center_x, center_y, radius = derive_roundabout_geometry(df)
        print(f"Center=({center_x:.2f}, {center_y:.2f}), radius={radius:.2f}m")
    except ValueError as e:
        print(f"Error in geometry: {e}")
        return
    
    features_df = prepare_features(df, center_x, center_y, radius)
    if features_df.empty:
        print("No valid trajectories after feature extraction")
        return
    
    numeric_features = [
        'curvature_adherence', 'lane_changes', 'forbidden_transitions',
        'mean_speed', 'speed_variance', 'accel_variance',
        'is_stationary', 'anti_clockwise', 'corner_proximity',
        'path_length', 'angle_variance'
    ]
    
    X = features_df[numeric_features].values
    y = features_df['label'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_scaled, y = smote.fit_resample(X_scaled, y)
    print(f"After SMOTE: {len(y)} samples, {sum(y==0)} normal, {sum(y==1)} abnormal")
    
    rf_clf = RandomForestClassifier(
        random_state=42,
        n_estimators=400,
        max_depth=20,
        class_weight={0:1, 1:2}
    )
    
    # Use 10-fold to get folds 3-7
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_reports = []
    train_indices = []
    test_indices = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        if fold in [2, 3, 4, 5, 6]:  # Folds 3-7 (0-based: 2-6)
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            rf_clf.fit(X_train, y_train)
            y_pred = rf_clf.predict(X_test)
            accuracy = (y_pred == y_test).mean()
            fold_accuracies.append(accuracy)
            fold_reports.append(classification_report(y_test, y_pred, output_dict=True))
            train_indices.extend(train_idx)
            test_indices.extend(test_idx)
            print(f"\nFold {fold+1} Accuracy: {accuracy:.4f}")
            print(f"Fold {fold+1} Classification Report:")
            print(classification_report(y_test, y_pred))
    
    # Compute average accuracy
    avg_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"\nAverage Accuracy (Folds 3-7): {avg_accuracy:.4f} (±{std_accuracy:.4f})")
    
    # Train final model on combined folds 3-7
    train_indices = np.unique(train_indices)
    test_indices = np.unique([i for i in test_indices if i not in train_indices])
    X_train, y_train = X_scaled[train_indices], y[train_indices]
    X_test, y_test = X_scaled[test_indices], y[test_indices]
    
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    final_accuracy = (y_pred == y_test).mean()
    final_report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\nFinal Model Accuracy (Folds 3-7 Combined): {final_accuracy:.4f}")
    print(f"Final Model Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model, scaler, and geometry
    joblib.dump(rf_clf, os.path.join(MODEL_DIR, 'rf_model_fold3to7.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump((center_x, center_y, radius), os.path.join(MODEL_DIR, 'geometry.pkl'))
    print(f"Model, scaler, and geometry saved to {MODEL_DIR}")
    
    # Evaluate on full data for consistency
    y_pred_full = rf_clf.predict(X_scaled)
    print(f"\nFull Data Confusion Matrix:")
    print(confusion_matrix(y, y_pred_full))
    
    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"Average Accuracy (Folds 3-7): {avg_accuracy:.4f} (±{std_accuracy:.4f})\n")
        f.write("\nFinal Model Accuracy (Folds 3-7 Combined): {final_accuracy:.4f}\n")
        f.write("\nFinal Model Classification Report:\n")
        for cls in ['0', '1']:
            f.write(f"Class {cls}: Precision={final_report[cls]['precision']:.2f}, "
                    f"Recall={final_report[cls]['recall']:.2f}, F1={final_report[cls]['f1-score']:.2f}\n")
        f.write("\nFold Accuracies:\n")
        for i, acc in enumerate(fold_accuracies, 3):
            f.write(f"Fold {i}: {acc:.4f}\n")
        f.write("\nFull Data Confusion Matrix:\n")
        f.write(str(confusion_matrix(y, y_pred_full)))
    
    plot_trajectories(df, center_x, center_y, radius, y_pred_full, features_df)
    
    importances = pd.DataFrame({
        'feature': numeric_features,
        'importance': rf_clf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(f"\nFeature Importances:")
    print(importances)
    importances.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

if __name__ == "__main__":
    root_dir = "interpolated"
    print("Running train_model.py with root_dir:", root_dir)
    main(root_dir)