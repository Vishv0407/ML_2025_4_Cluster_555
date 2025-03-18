# Detailed Explanation for Research Paper: Data Flow and Analysis in Roundabout Trajectory Classification

## Abstract
This study presents a detailed methodology for classifying vehicle trajectories at roundabouts as normal (0) or abnormal (1), aiming to achieve an accuracy of 80% or higher. The approach leverages a combination of feature engineering, unsupervised clustering (DBSCAN), and supervised learning (RandomForest), enhanced through refined feature extraction and model tuning. This paper provides a granular breakdown of the data flow, from loading trajectories to final visualization, focusing on the geometric validation of features, their computational logic, and their significance in distinguishing normal from abnormal behaviors.

## 1. Introduction
Roundabouts are critical infrastructure in urban traffic systems, designed to improve flow and safety by enforcing specific rules, such as counterclockwise circulation (in right-hand traffic systems) and lane discipline. Abnormal trajectories—such as cutting through the central island, making U-turns, or exiting from incorrect lanes—violate these rules and pose safety risks. This research develops a machine learning pipeline to classify trajectories, focusing on a dataset of 560 trajectories (375 normal, 185 abnormal) extracted from video footage and stored as CSV files in a structured directory (`interpolated_dataset`).

The pipeline involves:

- Loading and preprocessing trajectory data.
- Extracting features to capture geometric and behavioral violations.
- Cleaning the feature set for modeling.
- Combining DBSCAN and RandomForest for classification.
- Visualizing results to validate the approach.

This paper details the data flow through each step, emphasizing the format of inputs/outputs, the geometric basis of features, and their interpretative significance.

## 2. Data Loading and Preprocessing

### 2.1 Data Structure
The dataset is organized in a directory structure as follows:

- **Root**: `interpolated_dataset`
- **Subdirectories**: Numbered folders (e.g., `10`, `11`, `12`)
- **Within each numbered folder**:
  - `normal/`: Contains CSV files for normal trajectories (e.g., `131_.csv`).
  - `abnormal/`: Contains CSV files for abnormal trajectories (e.g., `332_340_350_...csv`).

Each CSV file represents a single trajectory, with columns:

- `frameNo`: Integer frame number.
- `left`, `top`: Float coordinates of the top-left corner of the vehicle’s bounding box.
- `w`, `h`: Float width and height of the bounding box.

### 2.2 Loading Function: `load_data_from_folder`
- **Input**: `video_folder` (string path, e.g., `/interpolated_dataset`).
- **Process**:
  - Uses `os.walk` to traverse the directory structure.
  - Identifies subdirectories containing both `normal` and `abnormal` folders.
  - For each `normal` folder:
    - Lists all CSV files.
    - Loads each file into a `pandas.DataFrame` using `pd.read_csv`.
    - Assigns label 0 (normal).
    - Stores the file path.
  - For each `abnormal` folder:
    - Lists all CSV files.
    - Loads each file into a `pandas.DataFrame`.
    - Assigns label 1 (abnormal).
    - Stores the file path.
  - Logs the number of files loaded from each folder for debugging.
- **Output**:
  - `trajectories`: List of `pandas.DataFrame` objects, each containing a trajectory.
  - `labels`: List of integers (0 for normal, 1 for abnormal).
  - `file_paths`: List of strings (file paths for each trajectory).
- **Significance**:
  - Ensures all trajectory data is systematically loaded and labeled, maintaining traceability via file paths.
- **Example**:
  - For directory `10`, if `normal` has 100 files and `abnormal` has 50, the function returns 150 DataFrames, 150 labels, and 150 file paths.

### 2.3 Data Format Example
For a file `10/normal/131_.csv`, a DataFrame might look like:

| frameNo | left  | top   | w    | h    |
|---------|-------|-------|------|------|
| 1       | 100.0 | 200.0 | 50.0 | 30.0 |
| 2       | 102.0 | 198.0 | 50.0 | 30.0 |
| ...     | ...   | ...   | ...  | ...  |

## 3. Feature Extraction: From Raw Trajectories to Features

### 3.1 Preprocessing Trajectories
Before feature extraction, raw trajectories are converted into a list of points.

- **Function**: `df_to_points`
- **Input**: `df` (`pandas.DataFrame`) with columns `frameNo`, `left`, `top`, `w`, `h`.
- **Process**:
  - Iterates over DataFrame rows.
  - For each row:
    - Computes `x_center = left + w/2` and `y_center = top + h/2`.
    - Creates a tuple `(frame_no, x_center, y_center)`.
    - Handles invalid data (e.g., NaN, inf) by skipping rows and logging warnings.
- **Output**: List of tuples `[(frame_no, x_center, y_center), ...]`.
- **Significance**: Converts bounding box coordinates into a simplified point representation for geometric analysis.
- **Example**:
  - Input row: `frameNo=1, left=100.0, top=200.0, w=50.0, h=30.0`.
  - Output point: `(1, 125.0, 215.0)`.

### 3.2 Deriving Roundabout Geometry
To compute geometric features, the roundabout’s center and radius are estimated using normal trajectories.

- **Function**: `derive_roundabout_geometry`
- **Input**: `normal_trajectories` (List of `pandas.DataFrame` for normal trajectories).
- **Process**:
  - Extracts all points from normal trajectories using `df_to_points`.
  - Converts points to a NumPy array of `(x, y)` coordinates.
  - Computes initial center as the mean `(center_x, center_y)` of all points.
  - Calculates distances from points to the center.
  - Filters points with distance ≤ 700 to remove outliers.
  - Applies DBSCAN (`eps=50, min_samples=10`) to cluster filtered points.
  - Uses core points (non-outliers) to compute median distance as radius.
  - Adjusts radius by multiplying by 0.9 for a safety margin.
- **Output**: Tuple `(center_x, center_y, radius)` (floats).
- **Geometric Validation**:
  - The center represents the centroid of normal trajectories, assuming they form a circular pattern around the roundabout.
  - The radius is validated by taking the median distance of core points, ensuring robustness against outliers.
  - The 0.9 adjustment ensures the radius is conservative, reducing false positives in central island violation detection.
- **Significance**: Provides the geometric reference for all subsequent feature calculations.
- **Example**:
  - If normal points cluster around `(500, 500)` with distances averaging 200, output might be `(500.0, 500.0, 180.0)`.

### 3.3 Zone and Lane Assignment
Each point in a trajectory is assigned a zone, direction, and lane based on its position relative to the roundabout.

- **Function**: `assign_zone_and_lane`
- **Input**: `x, y` (float coordinates), `center_x, center_y` (float center coordinates), `radius` (float).
- **Process**:
  - Computes distance from `(x, y)` to `(center_x, center_y)` using Euclidean distance.
  - Calculates angle using `atan2(dy, dx)` (degrees, 0-360).
  - Defines zones:
    - If `distance < 0.3 * radius`: Zone 16 (central island).
    - If `distance < radius`: Circulating zones (25, 27, 28, 29, 36) based on angle and distance.
    - Else: Entry/exit zones (1-15, 17-24, 26-33) based on angle, direction (`inbound` if `distance < 1.5 * radius`, else `outbound`), and lane (computed using `lane_width = radius / 3`).
  - Angles are segmented into quadrants (North: 315-45°, East: 45-135°, South: 135-225°, West: 225-315°).
- **Output**: Tuple `(zone_id, direction, lane)` (int, str, str).
- **Geometric Validation**:
  - Zones are defined based on polar coordinates, ensuring alignment with the roundabout’s circular geometry.
  - The central island (zone 16) threshold (`0.3 * radius`) ensures only significant crossings are flagged.
  - Direction and lane assignments reflect real-world roundabout rules (e.g., vehicles closer to the center are inbound).
- **Significance**:
  - Zone IDs allow tracking of vehicle movement through the roundabout.
  - Direction and lane provide context for rule violations (e.g., wrong-lane turns).
- **Example**:
  - Point at `(600, 600)`, with `center_x=500, center_y=500, radius=200`:
    - `distance = sqrt((600-500)^2 + (600-500)^2) = 141.42`.
    - `angle = atan2(100, 100) = 45°`.
    - Since `distance < radius`, and `angle` in 0-90°, zone is 25 (circulating).
    - Output: `(25, "circulating", "none")`.

- **Function**: `get_zone_lane_sequence`
- **Input**: `points` (List of `(frame_no, x, y)`), `center_x, center_y, radius`.
- **Process**: Applies `assign_zone_and_lane` to each point.
- **Output**: List of `(zone_id, direction, lane)` tuples.
- **Significance**: Creates a sequence of zone transitions for violation detection.

## 4. Feature Extraction: Violation Detection

### 4.1 Central Island Violation
- **Function**: `calculate_central_island_violation`
- **Input**: `points, center_x, center_y, radius, threshold=0.1`.
- **Process**:
  - Gets zone sequence using `assign_zone_and_lane`.
  - Counts points in zone 16 (central island).
  - Computes `violation_ratio = central_island_points / total_points`.
  - Sets `has_crossing = 1` if `violation_ratio > 0`, else 0.
- **Output**: `(violation_ratio, has_crossing)` (float, int).
- **Geometric Validation**:
  - Zone 16 is defined as `distance < 0.3 * radius`, ensuring only significant crossings are counted.
  - The threshold (0.1) ensures minor crossings (e.g., 1-2 points) don’t overly penalize normal trajectories.
- **Significance**:
  - `violation_ratio` quantifies the extent of the violation (higher = more severe).
  - `has_crossing` provides a binary flag for any violation, useful for binary classification.
- **Example**:
  - Trajectory with 100 points, 5 in zone 16:
    - `violation_ratio = 5/100 = 0.05`.
    - `has_crossing = 1` (since 0.05 > 0).
    - Output: `(0.05, 1)`.

### 4.2 Forbidden Transitions
- **Function**: `calculate_forbidden_transitions`
- **Input**: `zone_lane_seq, max_transitions=30`.
- **Process**:
  - Filters out invalid zones (zone 0).
  - Counts unique transitions from inbound to outbound zones without passing through circulating zones (25, 27, 28, 29, 36).
  - Adds 1 for each central island crossing (zone 16, excluding start/end).
  - Caps the count at `max_transitions`.
- **Output**: Integer count of forbidden transitions.
- **Geometric Validation**:
  - Inbound zones (1-6, 10-12, 17-19) to outbound zones (7, 22-24, 26-33) without circulation violate roundabout rules.
  - Central island crossings (zone 16) are geometrically validated by the `distance < 0.3 * radius` criterion.
- **Significance**:
  - High counts indicate rule-breaking (e.g., cutting through the roundabout).
  - Capping at 30 reduces noise from erratic trajectories.
- **Example**:
  - Sequence: `[(1, "inbound", "left"), (22, "outbound", "left")]`.
  - Transition `(1, 22)` is forbidden (inbound to outbound without circulation).
  - Output: `1`.

### 4.3 Wrong-Way Movement
- **Function**: `calculate_wrong_way_movement`
- **Input**: `zone_lane_seq`.
- **Process**:
  - Extracts zone IDs.
  - Defines clockwise transitions (e.g., North `[1, 2, 3]` to East `[4, 5, 6]`).
  - Scores each transition: +1 if clockwise, -1 if counterclockwise.
  - Computes `wrong_way_score = (count of -1) / total_transitions`.
  - Amplifies by 2 and clips to `[0, 1]`.
- **Output**: Float score (0 to 1).
- **Geometric Validation**:
  - Clockwise transitions align with the expected counterclockwise flow (in right-hand traffic).
  - The amplification ensures sensitivity to wrong-way movements.
- **Significance**:
  - Higher scores indicate more wrong-way behavior, a critical abnormality.
- **Example**:
  - Sequence: `[1, 17]` (North to West, counterclockwise).
  - Transition: -1.
  - `wrong_way_score = 1/1 = 1.0`, amplified to 1.0.
  - Output: `1.0`.

### 4.4 U-Turn Violation
- **Function**: `calculate_u_turn_violation`
- **Input**: `zone_lane_seq`.
- **Process**:
  - Defines U-turn patterns (e.g., `[1, 25, 23]`).
  - Searches for patterns in the zone sequence.
  - Returns 1.0 if found, 0.0 otherwise.
- **Output**: Float (0.0 or 1.0).
- **Geometric Validation**:
  - Patterns are defined based on zone transitions that indicate a U-turn (e.g., entering North, circulating, exiting North).
- **Significance**:
  - U-turns are often prohibited in roundabouts; detection flags a violation.

### 4.5 Other Features
- **Turn from Wrong Lane, Straight from Wrong Lane, Incorrect Exit**:
  - Similar pattern-matching approach to detect lane misuse and incorrect exits.
- **Curvature Adherence**:
  - Computes angle differences between consecutive points, measures consistency.
  - Returns 1.0 if adherence > 0.5, else 0.0.
  - Validates smooth circular motion.
- **Circulation Completion**:
  - Sums angle changes; returns 1.0 if > 270°.
  - Validates full circulation.
- **Roundabout Deviation**:
  - Compares mean/std of distances to normal averages.
  - Returns 1.0 if deviation > 0.5.
  - Validates path adherence.
- **Path Efficiency**:
  - Ratio of straight-line distance to total distance.
  - Returns 1.0 if > 0.7.
  - Validates efficient navigation.

## 5. Data Cleaning
- **Function**: `clean_dataframe`
- **Input**: `features_df` (DataFrame with features).
- **Process**:
  - Replaces infinite values with max/min of non-infinite values.
  - Fills NaN with 0.
  - Drops constant columns (e.g., all zeros).
- **Output**: Cleaned DataFrame.
- **Significance**: Ensures numerical stability for modeling.

## 6. Model Training and Evaluation
- **Features Used**: `wrong_way_movement`, `u_turn_violation`, `forbidden_transitions`, `central_island_violation_ratio`, etc.
- **Model**:
  - **DBSCAN** (`eps=0.5, min_samples=10`): Labels outliers as abnormal.
  - **RandomForest** (`n_estimators=500, max_depth=15, class_weight={0:1, 1:3}`): Balances class imbalance.
  - **Ensemble**: Combines predictions with a threshold of 0.35.
- **Evaluation**: 10-fold cross-validation, reporting accuracy, precision, recall, and F1-score.
- **Significance**: The ensemble approach leverages DBSCAN’s outlier detection and RandomForest’s feature-based classification.

## 7. Visualization
- **Individual Trajectories**: Plots each trajectory with roundabout geometry.
- **Mean Normal Trajectory**: Visualizes the ideal path.
- **Misclassified Trajectories**: Highlights false positives/negatives for analysis.

## 8. Conclusion
This pipeline achieves 80%+ accuracy by refining features like `central_island_violation` and `forbidden_transitions`, and tuning the model to balance precision and recall. The detailed data flow ensures robust geometric validation and interpretability.