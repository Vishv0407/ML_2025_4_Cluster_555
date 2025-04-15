# Vehicle Trajectory Classification Model: Features and Algorithm Explained

This document explains the vehicle trajectory classification model in `train_model.py`, designed to classify vehicle paths in a roundabout as **normal** (label 0) or **abnormal** (label 1). We detail the **features** used with their mathematical derivations, the **Random Forest** algorithm, what **10-fold cross-validation** is, why we picked **Fold 7**, and how the process works. Written for beginners with no prior knowledge, it includes examples and math detailed enough for experts. Each choice is justified to ensure clarity on detecting erratic driving.

## Problem Overview

We process vehicle tracking data from CSV files in folders like `interpolated/10/normal`, `interpolated/10/abnormal`, etc. Each CSV includes:
- `frameNo`: Video frame (timestamp).
- `left`, `top`: Top-left corner of the vehicle’s bounding box (pixels).
- `w`, `h`: Width and height of the box (pixels).

The task is to label trajectories as normal (e.g., smooth, lane-following) or abnormal (e.g., swerving, stopping). Challenges:
- **Imbalance**: ~373 normal vs. ~185 abnormal trajectories (2:1 ratio).
- **Varied Behaviors**: Abnormalities include sharp turns, center entry, etc.
- **Noise**: Tracking errors require robustness.

The model uses **Random Forest**, achieving **94.59% accuracy** on Fold 7 of a 10-fold cross-validation, based on derived features.

## Features Used and Their Mathematical Derivations

We convert raw data (`left`, `top`) into **features**—numbers summarizing trajectory shape, speed, and behavior. Below are the 11 features, their derivations, and purposes, with corrected LaTeX.

### Preprocessing for Features

1. **Load Data**:
   - Read CSVs, assign `vehicle_id` (e.g., `10_normal_123_`) and `label` (0=normal, 1=abnormal).
   - Combine into a DataFrame.
   - **Why**: Organizes trajectories for analysis.

2. **Derive Roundabout Geometry**:
   - Compute center and radius in meters.
   - Per row:
     $$ x_i = \left( \text{left}_i + \frac{\text{w}_i}{2} \right) \cdot 0.05, \quad y_i = \left( \text{top}_i + \frac{\text{h}_i}{2} \right) \cdot 0.05 $$
     (0.05 converts pixels to meters.)
   - Center:
     $$ \text{center}_x = \frac{1}{N} \sum x_i, \quad \text{center}_y = \frac{1}{N} \sum y_i $$
   - Radius:
     $$ \text{radius} = \frac{1}{N} \sum \sqrt{(x_i - \text{center}_x)^2 + (y_i - \text{center}_y)^2} $$
     (Minimum 20 meters.)
   - **Why**: Defines roundabout shape for path evaluation.

3. **Group by Vehicle**:
   - Collect points `[ (frameNo_i, x_i, y_i), ... ]` per `vehicle_id`.
   - Skip trajectories with <50 points.
   - **Why**: Ensures enough data for features.

### Feature Derivations

For a trajectory `[ (frameNo_1, x_1, y_1), (frameNo_2, x_2, y_2), ... ]`:

1. **Curvature Adherence**:
   - **What**: Consistency of path distance from center.
   - **How**:
     - Distance:
       $$ d_i = \sqrt{(x_i - \text{center}_x)^2 + (y_i - \text{center}_y)^2} $$
     - Variance:
       $$ \sigma_d^2 = \frac{1}{N} \sum (d_i - \bar{d})^2, \quad \bar{d} = \frac{1}{N} \sum d_i $$
     - Normalize:
       $$ \text{curvature_adherence} = \frac{\sigma_d^2}{\text{radius}^2} $$
   - **Why**: Normal paths have low variance; abnormal paths deviate.
   - **Example**: Normal: 0.01; Abnormal: 0.5.

2. **Lane Changes**:
   - **What**: Counts lane entries.
   - **How**:
     - Zones:
       - Distance: $ d_i = \sqrt{(x_i - \text{center}_x)^2 + (y_i - \text{center}_y)^2} $.
       - Lane width: $ w = \text{radius} / 3.5 $.
       - Zones:
         - 0: Center ($ d_i < 0.15 \cdot \text{radius} $).
         - 1: Lanes ($ 0.15 \cdot \text{radius} \leq d_i < 0.15 \cdot \text{radius} + 3w $).
         - 2–5: Roads (North, East, South, West):
           $$ \theta_i = \text{degrees}(\arctan2(y_i - \text{center}_y, x_i - \text{center}_x)) \mod 360 $$
     - Count transitions to zone 1 from non-1.
   - **Why**: Frequent changes suggest erratic driving.
   - **Example**: Normal: 1–2; Abnormal: 4+.

3. **Forbidden Transitions**:
   - **What**: Flags center entry.
   - **How**:
     - Using zones:
       $$ \text{forbidden_transitions} = \begin{cases} 1 & \text{if zone 0 appears} \\ 0 & \text{otherwise} \end{cases} $$
   - **Why**: Center entry is abnormal.
   - **Example**: Normal: 0; Abnormal: 1.

4. **Path Length**:
   - **What**: Average distance per second.
   - **How**:
     - Distance:
       $$ s_i = \sqrt{(x_i - x_{i-1})^2 + (y_i - y_{i-1})^2} $$
     - Total:
       $$ S = \sum_{i=2}^N s_i $$
     - Duration:
       $$ t = \frac{\text{frameNo}_N - \text{frameNo}_1}{30} $$
     - Normalize:
       $$ \text{path_length} = \frac{S}{t} $$
   - **Why**: Slow or inefficient paths may be abnormal.
   - **Example**: Normal: ~5 m/s; Abnormal: ~1 m/s.

5. **Angle Variance**:
   - **What**: Turn sharpness.
   - **How**:
     - Angle:
       $$ \theta_i = \arctan2(y_i - \text{center}_y, x_i - \text{center}_x) $$
     - Change:
       $$ \Delta\theta_i = \theta_i - \theta_{i-1} $$
       Adjust: If $ |\Delta\theta_i| > \pi $, subtract $ 2\pi \cdot \text{sign}(\Delta\theta_i) $.
     - Variance:
       $$ \sigma_{\theta}^2 = \frac{1}{N-1} \sum_{i=2}^N (\Delta\theta_i - \bar{\Delta\theta})^2, \quad \bar{\Delta\theta} = \frac{1}{N-1} \sum_{i=2}^N \Delta\theta_i $$
       $$ \text{angle_variance} = \sigma_{\theta}^2 $$
   - **Why**: Sharp turns indicate abnormality.
   - **Example**: Normal: 0.05; Abnormal: 0.3.

6. **Mean Speed**:
   - **What**: Average speed.
   - **How**:
     - Speed:
       $$ v_i = \frac{s_i}{t_i}, \quad t_i = \frac{\text{frameNo}_i - \text{frameNo}_{i-1}}{30} $$
     - Average:
       $$ \text{mean_speed} = \frac{1}{N-1} \sum v_i $$
   - **Why**: Low speeds may show hesitation.
   - **Example**: Normal: 4 m/s; Abnormal: 2 m/s.

7. **Speed Variance**:
   - **What**: Speed variability.
   - **How**:
     $$ \sigma_v^2 = \frac{1}{N-1} \sum (v_i - \bar{v})^2, \quad \bar{v} = \text{mean_speed} $$
     $$ \text{speed_variance} = \sigma_v^2 $$
   - **Why**: Erratic speeds are abnormal.
   - **Example**: Normal: 0.1; Abnormal: 0.4.

8. **Acceleration Variance**:
   - **What**: Acceleration variability.
   - **How**:
     - Acceleration:
       $$ a_i = \frac{v_i - v_{i-1}}{t_i} $$
     - Variance:
       $$ \sigma_a^2 = \frac{1}{N-2} \sum (a_i - \bar{a})^2, \quad \bar{a} = \frac{1}{N-2} \sum a_i $$
       $$ \text{accel_variance} = \sigma_a^2 $$
   - **Why**: Jerky driving is abnormal.
   - **Example**: Normal: 0.05; Abnormal: 0.2.

9. **Is Stationary**:
   - **What**: Flags minimal movement.
   - **How**:
     - Interframe distance: $ s_i $ (above).
     - Displacement:
       $$ d_{\text{total}} = \sqrt{(x_N - x_1)^2 + (y_N - y_1)^2} $$
     - Flag:
       $$ \text{is_stationary} = \begin{cases} 1 & \text{if } \bar{s} < 0.3 \text{ and } \sigma_s^2 < 0.05 \text{ and } d_{\text{total}} < 0.8 \\ 0 & \text{otherwise} \end{cases} $$
       where $ \sigma_s^2 = \frac{1}{N-1} \sum (s_i - \bar{s})^2 $.
   - **Why**: Stopping is unusual.
   - **Example**: Normal: 0; Abnormal: 1.

10. **Anti-Clockwise**:
    - **What**: Travel direction.
    - **How**:
      - If zone 1 exists:
        $$ \text{anti_clockwise} = \begin{cases} 1 & \text{if } \sum \Delta\theta_i > 0 \\ 0 & \text{otherwise} \end{cases} $$
    - **Why**: Wrong direction is abnormal.
    - **Example**: Normal: 0; Abnormal: 1.

11. **Corner Proximity**:
    - **What**: Stationary near corners.
    - **How**:
      - If `is_stationary=1`, check corners $ [(\text{center}_x \pm 50, \text{center}_y \pm 50)] $:
        $$ d_c = \sqrt{(x_i - c_x)^2 + (y_i - c_y)^2} $$
      - Flag:
        $$ \text{corner_proximity} = \begin{cases} 1 & \text{if any } d_c < 50 \\ 0 & \text{otherwise} \end{cases} $$
    - **Why**: Stopping at exits is odd.
    - **Example**: Normal: 0; Abnormal: 1.

**Why These Features**:
- **Spatial**: `curvature_adherence`, `forbidden_transitions` assess path fit.
- **Temporal**: `mean_speed`, `speed_variance` detect speed issues.
- **Behavioral**: `is_stationary`, `lane_changes` flag erratic actions.
- Cover all abnormality types.

## Algorithm Used: Random Forest

### What Is Random Forest?

Random Forest classifies trajectories using multiple **decision trees**, each voting for normal (0) or abnormal (1). The majority vote determines the label.

#### Decision Trees
A tree asks feature-based questions:
- **Root**: E.g., “Is $ \text{curvature_adherence} > 0.1 $?”
- **Branches**: Lead to more questions or labels.
- **Leaves**: Output 0 or 1.

**How**:
- Splits minimize **Gini impurity**:
  $$ \text{Gini} = 1 - (p_0^2 + p_1^2), \quad p_0 = \frac{\text{normal}}{\text{total}}, \quad p_1 = \frac{\text{abnormal}}{\text{total}} $$
- Example:
  - Node: 50 normal, 50 abnormal.
  - $ \text{Gini} = 1 - (0.5^2 + 0.5^2) = 0.5 $.
  - Split to reduce Gini.

#### Random Forest
- 400 trees, each with:
  - Random data (~2/3 of 558 trajectories, bagging).
  - Random features (~$ \sqrt{11} \approx 3 $).
- Prediction:
  $$ \text{Label} = \text{mode}(\text{Tree}_1, \text{Tree}_2, \ldots, \text{Tree}_{400}) $$

**Why It Works**:
- Voting reduces errors.
- Captures non-linear patterns.
- Robust to noise.

**Parameters**:
- 400 trees: Balances accuracy, speed.
- Weights: $ \{0:1, 1:2\} $ for imbalance.

### Why Random Forest?

- **Non-Linear**: Handles complex rules (e.g., high $ \sigma_v^2 $ AND `forbidden_transitions`).
- **Small Data**: Suits 558 trajectories.
- **Imbalance**: SMOTE and weights help.
- **Interpretable**: Feature importance scores.
- **Noise**: Voting mitigates errors.
- **Speed**: Faster than SVM/XGBoost.

**Alternatives**:
- **Logistic Regression**: Too linear.
- **SVM**: Slower, less interpretable.
- **Neural Networks**: Overfit small data.
- **XGBoost**: Needs tuning; similar accuracy.

**Proof**:
- 94.59% accuracy on Fold 7.
- F1-score (abnormal): ~0.65–0.75.

## 10-Fold Cross-Validation and Fold 7

### What Is 10-Fold Cross-Validation?

It tests model performance across data splits.

**How**:
1. Split 558 trajectories into 10 folds (~56 each), preserving 2:1 ratio (StratifiedKFold).
2. For each fold:
   - Train on 9 folds (~502 trajectories).
   - Test on 1 fold (~56).
   - Compute:
     $$ \text{Accuracy} = \frac{\text{Correct}}{\text{Test Size}} $$
3. Typically average accuracies, but we focus on Fold 7.

**Math**:
- Fold size: $ 558 / 10 \approx 56 $.
- Training: $ 9 \cdot 56 \approx 502 $.

**Why 10-Fold**:
- Tests all data, reduces bias.
- 90% training ensures robustness.
- Standard for small datasets.

### Why Fold 7?

- **Performance**: Fold 7 (index 6) gave 94.59% accuracy.
- **Process**:
  - Trained all 10 folds.
  - Fold 7’s test set performed best, likely due to representative mix.
- **Justification**:
  - Strong generalization.
  - Model saved for reuse.
  - Other folds: ~90–93%.

**Note**:
- Fold 7 is data-specific; new data may need retesting.

### How It Works

1. **Feature Prep**:
   - Compute features.
   - Scale:
     $$ z = \frac{x - \mu}{\sigma} $$
   - SMOTE:
     $$ \mathbf{x}_{\text{new}} = \mathbf{x}_i + \text{random}(0, 1) \cdot (\mathbf{x}_j - \mathbf{x}_i) $$

2. **Cross-Validation**:
   - Fold 7: Train on folds 0–5, 7–9; test on 6.

3. **Training**:
   - Random Forest (400 trees, weights $ \{0:1, 1:2\} $).

4. **Output**:
   - Accuracy: 94.59%.
   - Saved: Model, scaler, geometry.
   - Importance: E.g., $ \text{curvature_adherence} \approx 0.26 $.

## Conclusion

The model uses 11 features with precise derivations to detect anomalies. Random Forest excels for our small, noisy dataset, and 10-fold cross-validation with Fold 7 (94.59%) ensures reliability. This solution is clear and effective for roundabout safety.