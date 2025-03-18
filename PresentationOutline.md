# Progress Presentation: Vehicle Trajectory Classification at Roundabouts

## Slide 1: Introduction
**Image Label:** [Image_01_Introduction.jpg]  
- **Course Name:** Machine Learning (6th Semester, Ahmedabad University)  
- **Project Name:** Vehicle Trajectory Classification at Roundabouts  
- **Group Details:**  
  - Group Leader: [Your Name]  
  - Members: [Member 1], [Member 2], [Member 3]  
  - Supervisor: [Supervisor Name]  
- **Content:** Welcome to our progress presentation on the Vehicle Trajectory Classification project. This project, part of our Machine Learning course, focuses on developing a system to classify vehicle trajectories at roundabouts as normal or abnormal. Our group, under the guidance of [Supervisor Name], has been working diligently to advance this research. Today, we’ll cover the problem statement, literature survey, dataset discussion, our approach to feature derivation, and future work, stopping at feature derivation without delving into classification results.

## Slide 2: Problem Statement
**Image Label:** [Image_02_ProblemStatement.jpg]  
- **Content:**  
  - Roundabouts are designed to enhance traffic flow and safety by enforcing rules like counterclockwise circulation and lane discipline.  
  - Abnormal trajectories, such as cutting through the central island or making U-turns, violate these rules and increase accident risks.  
  - The challenge is to develop a robust method to identify these abnormal behaviors using trajectory data extracted from video footage.  
  - Our research aims to create a feature-rich dataset and derive meaningful features to differentiate normal and abnormal trajectories effectively.  

# Literature Survey

## Slide 3: Literature Survey
**Image Label:** [Image_03_LiteratureSurvey.jpg]  
- **Content:**  
  - Below is a table summarizing recent research relevant to our project on trajectory classification, including the algorithms and methodologies used:

| **Paper Title**                              | **Authors**                                      | **Year** | **Algorithm and Methodology Used**            | **Key Contribution**                          | **Relevance to Our Work**                  |
|----------------------------------------------|-------------------------------------------------|----------|-----------------------------------------------|----------------------------------------------|--------------------------------------------|
| Vision-based investigation of road traffic and violations at urban roundabout in India using UAV video | Yagnik M Bhavsar et al.                        | 2023     | Zone-based detection with image processing and rule-based classification | Used UAV video to analyze traffic violations at roundabouts, focusing on zone-based detection. | Directly applies to our roundabout violation detection using video data. |
| Driver Profile and Driving Pattern Recognition for Road Safety Assessment | Dimitrios I. Tselentis and Eleonora Papadimitriou | 2023     | Statistical analysis and clustering (e.g., k-Means, hierarchical clustering) | Developed methods to recognize driving patterns for safety, using trajectory analysis. | Provides insights into pattern recognition applicable to our abnormal trajectory identification. |
| Driving Style Classification using Deep Temporal Clustering with Enhanced Explainability | Yuxiang Feng et al.                            | N/A      | Deep learning-based temporal clustering with convolutional neural networks (CNNs) and explainability techniques | Employed deep temporal clustering to classify driving styles with explainable features. | Offers a clustering approach that could enhance our feature derivation process. |
| Traffic Pattern Modeling, Trajectory Classification and Vehicle Tracking within Urban Intersections | Cheng-En Wu et al.                             | N/A      | Spatial-temporal modeling with Kalman filtering and pattern recognition techniques | Modeled traffic patterns and classified trajectories at intersections using spatial data. | Relevant for adapting intersection techniques to roundabout contexts. |
| Trajectory Data Classification: A Review      | Jiang Bian et al.                              | N/A      | Review of algorithms (e.g., k-Nearest Neighbors, Support Vector Machines, clustering methods) | Comprehensive review of trajectory classification methods and feature extraction techniques. | Serves as a foundational guide for selecting and deriving features in our project. |

## Slide 4: Dataset Discussion - Overview
**Image Label:** [Image_04_DatasetOverview.jpg]  
- **Content:**  
  - Our dataset comprises 560 vehicle trajectories extracted from video footage, stored as CSV files in a structured directory: `/home/run/media/localdiskD/Ahmedabad University/6th SEM/ML/ML_2025_4_Cluster_555/Codes/dataset_interpolated/processed/interpolated`.  
  - It is organized into subdirectories (10, 11, 12), each containing `normal` and `abnormal` folders.  
  - Each CSV file includes columns: `frameNo` (integer frame number), `left`, `top` (float coordinates of the top-left bounding box corner), `w`, `h` (float width and height of the bounding box).  
  - The dataset includes 375 normal trajectories and 185 abnormal trajectories, reflecting a 2:1 class imbalance.  

## Slide 5: Dataset Discussion - Distribution
**Image Label:** [Image_05_DatasetDistribution.jpg]  
- **Content:**  
  - **Directory 10:** 54 normal, 47 abnormal trajectories.  
  - **Directory 11:** 168 normal, 80 abnormal trajectories.  
  - **Directory 12:** 153 normal, 58 abnormal trajectories.  
  - This distribution shows variability across directories, with Directory 11 having the largest sample size, aiding in robust feature derivation.  
  - The abnormal trajectories include behaviors like central island crossings and U-turns, captured through video interpolation for consistent frame data.  

## Slide 6: Your Approach - Feature Derivation Process
**Image Label:** [Image_06_FeatureDerivation.jpg]  
- **Content:**  
  - **Step 1: Trajectory Preprocessing**  
    - Convert raw CSV data into a list of points using `df_to_points`, calculating the center of each vehicle’s bounding box (x_center = left + w/2, y_center = top + h/2).  
  - **Step 2: Roundabout Geometry Derivation**  
    - Use `derive_roundabout_geometry` to estimate the roundabout’s center and radius from normal trajectories, applying DBSCAN clustering to filter outliers and compute the median radius.  
  - **Step 3: Zone and Lane Assignment**  
    - Assign zones, directions, and lanes to each point using `assign_zone_and_lane`, based on distance and angle from the center, defining 36 zones including the central island (zone 16).  
  - **Step 4: Feature Extraction**  
    - Derive features like `central_island_violation` (ratio of points in zone 16), `forbidden_transitions` (count of invalid zone transitions), `wrong_way_movement` (score of counterclockwise violations), `u_turn_violation` (pattern matching for U-turns), `curvature_adherence` (angle consistency), `circulation_completion` (total angle change), `path_efficiency` (straight-line to total distance ratio), and `directional_variance` (variation in movement direction).  

## Slide 7: Your Approach - Future Work
**Image Label:** [Image_07_FutureWork.jpg]  
- **Content:**  
  - **Refine Feature Derivation:**  
    - Adjust zone thresholds (e.g., central island to 0.3 * radius) and add features like `lane_indiscipline` to capture lane misuse.  
  - **Enhance Geometry Estimation:**  
    - Improve `derive_roundabout_geometry` by tuning DBSCAN parameters (e.g., eps, min_samples) for better outlier removal.  
  - **Future Steps:**  
    - We plan to use DBSCAN and RandomForest as our classifiers to process these derived features, focusing on handling class imbalance and ensuring feature robustness before proceeding to model training and evaluation.  

## Slide 8: References
**Image Label:** [Image_08_References.jpg]  
- **Content:**  
  - Bhavsar, Y. M., et al. (2023). "Vision-based investigation of road traffic and violations at urban roundabout in India using UAV video." *Transportation Engineering*, 14, 100207.  
  - Tselentis, D. I., & Papadimitriou, E. (2023). "Driver Profile and Driving Pattern Recognition for Road Safety Assessment." *IEEE Open Journal of Intelligent Transportation Systems*, 10.1109/OJITS.2023.3237177.  
  - Feng, Y., et al. (N/A). "Driving Style Classification using Deep Temporal Clustering with Enhanced Explainability."  
  - Wu, C.-E., et al. (N/A). "Traffic Pattern Modeling, Trajectory Classification and Vehicle Tracking within Urban Intersections."  
  - Bian, J., et al. (N/A). "Trajectory Data Classification: A Review."  

---

# Speech Script for 20-Minute Presentation

## Introduction (2 minutes)
**Speech:**  
"Good [morning/afternoon], everyone. Welcome to our progress presentation on the Vehicle Trajectory Classification at Roundabouts project. We are part of the Machine Learning course at Ahmedabad University, 6th semester. Our project group consists of [Your Name] as the leader, along with [Member 1], [Member 2], and [Member 3], under the supervision of [Supervisor Name]. This project aims to classify vehicle trajectories at roundabouts as normal or abnormal using machine learning techniques. Today, we’ll walk you through the problem statement, a literature survey, details of our dataset, our approach to feature derivation, and our future work, focusing solely on the research progress up to feature derivation, without touching on classification results. This structured approach ensures we lay a solid foundation for the next phases of our project."

**Explanation:**  
- This opening sets the context, introduces the team, and outlines the presentation structure. Mentioning the course and supervision adds credibility. The focus on feature derivation aligns with the guideline, preparing the audience for a detailed research update. Background knowledge includes understanding that roundabouts are traffic systems requiring rule adherence, and trajectory classification is a growing field in traffic safety.

## Problem Statement (2 minutes)
**Speech:**  
"Roundabouts are key traffic infrastructure designed to improve flow and safety by enforcing rules like counterclockwise circulation and proper lane usage, especially in right-hand traffic systems. However, abnormal trajectories—such as vehicles cutting through the central island, making U-turns, or exiting from incorrect lanes—violate these rules and pose significant safety hazards. Our research addresses the challenge of developing a method to identify these abnormal behaviors using trajectory data extracted from video footage. The goal is to derive features that effectively capture these violations, providing a basis for future classification. This problem is critical as it impacts road safety and traffic management in urban areas."

**Explanation:**  
- This explains the importance of roundabouts and the specific violations we target, grounding the problem in real-world safety concerns. The mention of video data introduces the data source, a common method in traffic analysis. For Q&A, be ready to discuss why these violations matter (e.g., increased collision risk) and how video data is processed (e.g., frame-by-frame tracking).

## Literature Survey (3 minutes)
**Speech:**  
"Our literature survey highlights recent research relevant to our project. We’ve compiled a table of five key papers:  
1. *Vision-based investigation of road traffic and violations at urban roundabouts in India using UAV video* by Bhavsar et al. (2023) used UAV footage to detect violations at roundabouts, employing zone-based analysis—directly applicable to our video-based approach.  
2. *Driver Profile and Driving Pattern Recognition for Road Safety Assessment* by Tselentis and Papadimitriou (2023) analyzed driving patterns for safety, offering insights into trajectory-based pattern recognition for our work.  
3. *Driving Style Classification using Deep Temporal Clustering with Enhanced Explainability* by Feng et al. explored deep clustering for driving styles, which could inspire our feature derivation.  
4. *Traffic Pattern Modeling, Trajectory Classification and Vehicle Tracking within Urban Intersections* by Wu et al. modeled traffic at intersections, providing techniques adaptable to roundabouts.  
5. *Trajectory Data Classification: A Review* by Bian et al. reviewed classification methods, guiding our feature selection process. These studies collectively inform our approach by emphasizing video data, pattern recognition, and feature engineering."

**Explanation:**  
- This summarizes relevant research, linking each paper to our project’s context (e.g., video data, trajectory analysis). The table format aids clarity. For Q&A, explain how UAVs capture data (aerial perspective), the role of clustering in pattern recognition, and why reviews are foundational for methodology selection.

## Dataset Discussion - Overview (3 minutes)
**Speech:**  
"Our dataset consists of 560 vehicle trajectories derived from video footage, stored in CSV files within the directory `/home/run/media/localdiskD/Ahmedabad University/6th SEM/ML/ML_2025_4_Cluster_555/Codes/dataset_interpolated/processed/interpolated`. This directory is organized into subdirectories 10, 11, and 12, each containing `normal` and `abnormal` folders. Each CSV file includes `frameNo` for the frame number, `left` and `top` for the bounding box’s top-left corner coordinates, and `w` and `h` for width and height, all in float format. We have 375 normal trajectories and 185 abnormal ones, resulting in a 2:1 class imbalance. This dataset provides a rich basis for deriving features to distinguish trajectory behaviors."

**Explanation:**  
- This introduces the dataset’s structure and content, emphasizing the imbalance as a research consideration. For Q&A, be prepared to discuss how videos are converted to CSV (e.g., object detection algorithms), the significance of bounding box data (vehicle position tracking), and why imbalance matters (potential bias in analysis).

## Dataset Discussion - Distribution (3 minutes)
**Speech:**  
"Looking at the distribution, Directory 10 has 54 normal and 47 abnormal trajectories, Directory 11 has 168 normal and 80 abnormal, and Directory 12 has 153 normal and 58 abnormal. This variability across directories, with Directory 11 being the largest, helps ensure robust feature derivation. The abnormal trajectories include specific violations like central island crossings and U-turns, captured through video interpolation to maintain consistent frame data. This diversity and preprocessing step are crucial for our research, as they allow us to study a wide range of behaviors systematically."

**Explanation:**  
- This details the dataset’s spread, highlighting interpolation’s role in data consistency. For Q&A, explain interpolation (filling missing frames), why certain directories have more data (e.g., video length), and how violations are labeled (manual annotation or automated detection).

## Your Approach - Feature Derivation Process (4 minutes)
**Speech:**  
"Our approach to feature derivation involves several research steps:  
1. **Trajectory Preprocessing:** We use `df_to_points` to convert CSV data into a list of points, calculating the center of each vehicle’s bounding box as x_center = left + w/2 and y_center = top + h/2. This simplifies the data for geometric analysis.  
2. **Roundabout Geometry Derivation:** The `derive_roundabout_geometry` function estimates the roundabout’s center and radius using normal trajectories. We apply DBSCAN clustering to filter outliers and compute the median radius, providing a geometric reference.  
3. **Zone and Lane Assignment:** With `assign_zone_and_lane`, we assign each point a zone (1-36), direction (inbound/outbound), and lane (left/middle/right) based on its distance and angle from the center. Zone 16 represents the central island.  
4. **Feature Extraction:** We derive features such as `central_island_violation` (proportion of points in zone 16), `forbidden_transitions` (count of invalid zone transitions), `wrong_way_movement` (score of counterclockwise violations), `u_turn_violation` (pattern matching for U-turns), `curvature_adherence` (angle consistency between points), `circulation_completion` (total angle change), `path_efficiency` (ratio of straight-line to total distance), and `directional_variance` (variation in movement direction). These features capture the geometric and behavioral aspects of trajectories."

**Explanation:**  
- This outlines the research process, focusing on how features are conceptually derived. For Q&A, discuss DBSCAN’s role (clustering for outlier removal), zone assignment logic (polar coordinates), and feature definitions (e.g., why angle consistency matters for circulation). Avoid implementation details, emphasizing the research intent.

## Your Approach - Future Work (2 minutes)
**Speech:**  
"Looking ahead, we plan to refine our feature derivation. We’ll adjust zone thresholds, such as setting the central island to 0.3 times the radius, and introduce a new feature, `lane_indiscipline`, to detect lane misuse. We’ll also enhance `derive_roundabout_geometry` by tuning DBSCAN parameters like eps and min_samples for better outlier removal. Our next steps involve using DBSCAN and RandomForest as classifiers to process these features, focusing on handling class imbalance and ensuring feature robustness before proceeding further."

**Explanation:**  
- This sets future research goals, linking to the guideline’s focus on approach and future work. For Q&A, explain why tuning parameters matters (improves accuracy), what `lane_indiscipline` might entail (e.g., lane change violations), and why DBSCAN/RandomForest are chosen (unsupervised/supervised complementarity).

## References (1 minute)
**Speech:**  
"Finally, our references include: *Vision-based investigation of road traffic and violations at urban roundabout in India using UAV video* by Bhavsar et al. (2023), *Driver Profile and Driving Pattern Recognition for Road Safety Assessment* by Tselentis and Papadimitriou (2023), *Driving Style Classification using Deep Temporal Clustering with Enhanced Explainability* by Feng et al., *Traffic Pattern Modeling, Trajectory Classification and Vehicle Tracking within Urban Intersections* by Wu et al., and *Trajectory Data Classification: A Review* by Bian et al. These sources have guided our research direction."

**Explanation:**  
- This credits the literature, preparing for Q&A on how these papers influenced the methodology. Keep it brief, focusing on acknowledgment.

**Total Time:** ~20 minutes, with each section timed for clarity and Q&A readiness.

---

# Research Paper Context for Presentation

1. **Vision-based investigation of road traffic and violations at urban roundabout in India using UAV video (Bhavsar et al., 2023)**  
   - **Basic Work:** Utilized UAV-captured video to analyze traffic violations at roundabouts in India, employing a zone-based approach to detect rule-breaking behaviors like wrong-way driving.  
   - **How:** Processed video frames to extract vehicle trajectories, assigned zones based on spatial coordinates, and identified violations through pattern matching.  
   - **Context:** Directly aligns with our use of video-derived CSV data and zone assignment for violation detection, providing a practical framework for our feature derivation.

2. **Driver Profile and Driving Pattern Recognition for Road Safety Assessment (Tselentis & Papadimitriou, 2023)**  
   - **Basic Work:** Developed a method to recognize driving patterns using trajectory data, focusing on safety assessment through statistical and machine learning techniques.  
   - **How:** Analyzed multidimensional trajectory data to identify risky behaviors, using clustering to group similar patterns.  
   - **Context:** Offers a foundation for our pattern recognition approach, especially in deriving features like `wrong_way_movement` and `u_turn_violation`.

3. **Driving Style Classification using Deep Temporal Clustering with Enhanced Explainability (Feng et al.)**  
   - **Basic Work:** Proposed a deep temporal clustering method to classify driving styles, enhancing interpretability with feature importance analysis.  
   - **How:** Applied deep learning to time-series trajectory data, clustering styles and explaining results through feature contributions.  
   - **Context:** Inspires our use of clustering (via DBSCAN) for geometry derivation and feature extraction, suggesting potential for explainability in future phases.

4. **Traffic Pattern Modeling, Trajectory Classification and Vehicle Tracking within Urban Intersections (Wu et al.)**  
   - **Basic Work:** Modeled traffic patterns and classified trajectories at urban intersections using spatial and temporal data.  
   - **How:** Extracted features from vehicle tracks, modeled patterns with statistical methods, and tracked vehicles for classification.  
   - **Context:** Relevant for adapting intersection-based feature derivation (e.g., `path_efficiency`) to roundabout contexts in our research.

5. **Trajectory Data Classification: A Review (Bian et al.)**  
   - **Basic Work:** Provided a comprehensive review of trajectory classification methods, focusing on feature extraction and data preprocessing techniques.  
   - **How:** Surveyed approaches like clustering, pattern matching, and geometric analysis, offering a broad perspective on trajectory handling.  
   - **Context:** Serves as a theoretical guide for selecting and deriving our features, ensuring a well-rounded research approach.

These papers collectively support our focus on video-based trajectory analysis, zone-based feature derivation, and the use of clustering techniques, aligning with our research progress up to feature derivation.

# Detailed Explanation: DBSCAN, RandomForest, and Feature Derivation for Vehicle Trajectory Classification

## DBSCAN and RandomForest: Explanation, Appropriateness, and Comparison with Other Algorithms

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

#### What is DBSCAN?
DBSCAN is an unsupervised clustering algorithm that groups data points based on their density. It identifies clusters as regions of high density separated by regions of low density, marking points in sparse areas as outliers. DBSCAN operates with two key parameters:
- **eps (ε):** The maximum distance between two points for them to be considered part of the same cluster.
- **min_samples:** The minimum number of points required to form a dense region (a cluster).

The algorithm works as follows:
1. Start with an unvisited point, mark it as visited.
2. If the point has at least `min_samples` neighbors within `eps` distance, form a cluster.
3. Recursively expand the cluster by adding all density-reachable points.
4. Points not assigned to any cluster are labeled as outliers.

#### Why is DBSCAN Appropriate for Our Case?
- **Outlier Detection for Abnormal Trajectories:** In our project, abnormal trajectories (e.g., central island crossings, U-turns) often deviate significantly from normal patterns. DBSCAN excels at identifying such outliers without requiring the number of clusters to be predefined, unlike k-Means. This is critical because we don’t know how many distinct "normal" trajectory patterns exist.
- **Handling Non-Spherical Clusters:** Vehicle trajectories at roundabouts form complex, non-spherical patterns (e.g., circular paths). DBSCAN can handle such arbitrary shapes, making it suitable for clustering normal trajectories during the `derive_roundabout_geometry` step.
- **Feature Derivation Support:** We use DBSCAN to estimate the roundabout’s geometry by clustering normal trajectory points, filtering outliers to compute a reliable center and radius. This geometric foundation is essential for subsequent feature extraction like `central_island_violation` and `forbidden_transitions`.

#### Why Not Other Algorithms from the Literature Survey?
- **k-Shape (Paper 1 - Accelerating k-Shape Time Series Clustering):** k-Shape is a time-series clustering method that uses a shape-based distance metric, optimized for GPU acceleration. While it’s efficient for time-series data, our trajectories are spatial (x, y coordinates) rather than purely temporal. k-Shape assumes a fixed number of clusters and focuses on temporal alignment, which doesn’t suit our need for spatial outlier detection or geometry derivation.
- **k-Means (General Unsupervised Method):** k-Means requires specifying the number of clusters and assumes spherical clusters, which doesn’t fit our data’s complex, circular patterns. It also struggles with outliers, a critical aspect of our task, as it forces all points into clusters, potentially misclassifying abnormal trajectories.
- **DTW (Dynamic Time Warping, Paper 2 - Discovering Similar Multidimensional Trajectories):** DTW is a similarity measure for time-series data, often used to align trajectories with varying speeds. While DTW could help compare trajectory shapes, it’s computationally expensive for large datasets and doesn’t directly address outlier detection or spatial clustering, which DBSCAN handles efficiently in our case.
- **Deep Temporal Clustering (Paper 4 - Driving Style Classification):** This method uses deep learning for temporal clustering, requiring large datasets and computational resources. Our dataset (560 trajectories) is relatively small, and we prioritize interpretable features over black-box models at this stage.

### RandomForest

#### What is RandomForest?
RandomForest is a supervised ensemble learning algorithm that builds multiple decision trees and combines their predictions to improve accuracy and robustness. It operates as follows:
1. **Bootstrap Sampling:** Randomly sample the dataset with replacement to create multiple subsets.
2. **Feature Randomness:** At each node of a decision tree, select a random subset of features to split on, reducing correlation between trees.
3. **Tree Construction:** Build decision trees independently on each subset.
4. **Aggregation:** For classification, take a majority vote across all trees to predict the class.

RandomForest handles overfitting through averaging, provides feature importance scores, and is robust to noisy data.

#### Why is RandomForest Appropriate for Our Case?
- **Handling High-Dimensional Features:** Our derived features (e.g., `central_island_violation`, `path_efficiency`) create a high-dimensional feature space. RandomForest manages this effectively by selecting random subsets of features at each split, reducing overfitting.
- **Class Imbalance Management:** With 375 normal and 185 abnormal trajectories, our dataset has a 2:1 imbalance. RandomForest allows us to adjust class weights (e.g., higher weight for the abnormal class) to improve recall for abnormal trajectories, addressing this imbalance.
- **Feature Importance for Interpretability:** RandomForest provides feature importance scores, helping us understand which features (e.g., `forbidden_transitions`) are most discriminative for classifying normal vs. abnormal trajectories. This aligns with our research goal of creating interpretable features.
- **Robustness to Noise:** Trajectory data can be noisy due to video interpolation errors or tracking inaccuracies. RandomForest’s ensemble nature makes it resilient to such noise, ensuring reliable classification.

#### Why Not Other Algorithms from the Literature Survey?
- **Deep Temporal Clustering (Paper 4):** While effective for driving style classification, deep learning requires large datasets and significant computational resources. Our dataset is relatively small, and deep models lack the interpretability we need for understanding feature contributions.
- **Statistical Methods (Paper 3 - Driver Profile and Driving Pattern Recognition):** Tselentis et al. used statistical methods for pattern recognition, which are less effective for high-dimensional, non-linear data like ours. RandomForest captures complex relationships between features more effectively.
- **Intersection-Specific Methods (Paper 14 - Traffic Pattern Modeling):** Wu et al. used statistical modeling for intersections, which doesn’t generalize well to roundabouts due to different traffic dynamics. RandomForest’s flexibility makes it more suitable for our context.
- **Review-Based Methods (Paper 11 - Trajectory Data Classification: A Review):** Bian et al. discuss various methods (e.g., SVM, neural networks), but these often require extensive tuning or large datasets. RandomForest offers a balance of performance, ease of use, and interpretability for our needs.

### Why Not Unsupervised Classifiers Like k-Shape, k-Means, or DTW?

#### k-Shape
- **Why Not Used:** k-Shape is designed for time-series clustering, focusing on temporal alignment using a shape-based distance metric. Our trajectories are spatial (x, y coordinates over frames), not purely temporal, making k-Shape less relevant. Additionally, k-Shape requires specifying the number of clusters, which is impractical since we don’t know how many distinct trajectory patterns exist in our data.
- **Potential Use:** If we were to focus on temporal aspects (e.g., speed variations over time), k-Shape could be used to cluster trajectories by their temporal profiles, but this isn’t our primary focus.

#### k-Means
- **Why Not Used:** k-Means assumes spherical clusters and requires the number of clusters to be predefined, neither of which suits our data. Roundabout trajectories form non-spherical, circular patterns, and we don’t know how many clusters (normal patterns) exist. k-Means also doesn’t handle outliers well, which is critical for identifying abnormal trajectories.
- **Potential Use:** k-Means could be used as a baseline for clustering normal trajectories, but its limitations in handling outliers and non-spherical clusters make it less effective than DBSCAN for our geometry derivation step.

#### DTW (Dynamic Time Warping)
- **Why Not Used:** DTW measures similarity between time-series by aligning them temporally, which is useful for comparing trajectories with varying speeds. However, our focus is on spatial violations (e.g., central island crossings), not temporal alignment. DTW is also computationally expensive (O(n²) complexity), making it impractical for our dataset of 560 trajectories.
- **Potential Use:** DTW could be used to compare the shape of trajectories for similarity analysis, but it doesn’t directly address our need for outlier detection or feature-based classification.

### Why DBSCAN and RandomForest Together?
- **Complementary Strengths:** DBSCAN (unsupervised) excels at identifying outliers, which we use to detect potential abnormal trajectories and derive roundabout geometry. RandomForest (supervised) leverages labeled data to learn from derived features, providing robust classification and interpretability.
- **Pipeline Fit:** DBSCAN supports feature derivation (e.g., geometry estimation), while RandomForest will use these features for final classification, creating a hybrid approach that leverages both unsupervised and supervised learning.

---

## Feature Derivation: From Raw Data to Features

Below is a detailed explanation of how each feature mentioned in the presentation is derived from the raw data (CSV files containing `frameNo`, `left`, `top`, `w`, `h`).

### Step 1: Raw Data Description
- **Input Data:** Each CSV file represents a trajectory with columns:
  - `frameNo`: Integer frame number (e.g., 1, 2, 3, ...).
  - `left`, `top`: Float coordinates of the top-left corner of the vehicle’s bounding box.
  - `w`, `h`: Float width and height of the bounding box.
- **Example:** For `10/normal/131_.csv`, a sample row might be:


### Step 2: Trajectory Preprocessing (`df_to_points`)
- **Process:** Convert each row into a point representing the vehicle’s center.
- Compute `x_center = left + w/2` and `y_center = top + h/2`.
- Create a tuple `(frameNo, x_center, y_center)`.
- Skip rows with invalid data (e.g., NaN values).
- **Example:** For the row above:
- `x_center = 100.0 + 50.0/2 = 125.0`
- `y_center = 200.0 + 30.0/2 = 215.0`
- Output: `(1, 125.0, 215.0)`
- **Output:** A list of points, e.g., `[(1, 125.0, 215.0), (2, 126.0, 214.0), ...]`.

### Step 3: Roundabout Geometry Derivation (`derive_roundabout_geometry`)
- **Process:** Estimate the roundabout’s center and radius using normal trajectories.
- Collect all points from normal trajectories.
- Compute initial center as the mean of x and y coordinates.
- Filter points within 700 units of the center to remove outliers.
- Apply DBSCAN (eps=50, min_samples=10) to cluster filtered points.
- Use core points (non-outliers) to compute the final center and median radius, adjusted by a 0.9 factor for safety.
- **Example:** If points cluster around `(500, 500)` with a median distance of 200, the output might be `(500.0, 500.0, 180.0)`.

### Step 4: Zone and Lane Assignment (`assign_zone_and_lane`)
- **Process:** For each point `(frameNo, x, y)`, assign a zone, direction, and lane.
- Compute distance from `(x, y)` to the center using Euclidean distance: `sqrt((x - center_x)^2 + (y - center_y)^2)`.
- Compute angle using `atan2(y - center_y, x - center_x)` in degrees (0-360).
- Define zones:
  - If `distance < 0.3 * radius`: Zone 16 (central island).
  - If `distance < radius`: Circulating zones (25-36) based on angle and distance.
  - Else: Entry/exit zones (1-15, 17-24, 26-33) based on angle, direction (`inbound` if `distance < 1.5 * radius`, else `outbound`), and lane (computed using `lane_width = radius / 3`).
- **Example:** Point `(600, 600)`, center `(500, 500)`, radius `200`:
- `distance = sqrt((600-500)^2 + (600-500)^2) = 141.42`
- `angle = atan2(100, 100) = 45°`
- `distance < radius`, angle in 0-90°, so zone is 25 (circulating).
- Output: `(25, "circulating", "none")`.

### Step 5: Feature Extraction

#### `central_island_violation`
- **Process:** Calculate the proportion of points in zone 16 (central island).
- Count points where `zone == 16`.
- Compute `violation_ratio = central_island_points / total_points`.
- **Example:** Trajectory with 100 points, 5 in zone 16:
- `violation_ratio = 5/100 = 0.05`

#### `forbidden_transitions`
- **Process:** Count invalid transitions between zones (e.g., inbound to outbound without circulation).
- Iterate through the zone sequence, checking for transitions like `(1, 22)` (inbound to outbound without circulating zones like 25).
- Cap the count at 30 to reduce noise.
- **Example:** Sequence `[(1, "inbound", "left"), (22, "outbound", "left")]`: Transition `(1, 22)` is forbidden, count = 1.

#### `wrong_way_movement`
- **Process:** Score counterclockwise violations based on zone transitions.
- Define clockwise transitions (e.g., North `[1, 2, 3]` to East `[4, 5, 6]`).
- Score each transition: +1 if clockwise, -1 if counterclockwise.
- Compute `score = (count of -1) / total_transitions`, amplify by 2, clip to `[0, 1]`.
- **Example:** Sequence `[1, 17]` (North to West, counterclockwise): Score = 1.0.

#### `u_turn_violation`
- **Process:** Detect U-turn patterns in the zone sequence.
- Define patterns like `[1, 25, 23]` (enter North, circulate, exit North).
- Return 1.0 if a pattern is found, 0.0 otherwise.
- **Example:** Sequence `[1, 25, 23]`: U-turn detected, score = 1.0.

#### `curvature_adherence`
- **Process:** Measure angle consistency between consecutive points.
- For each triplet of points, compute the angle between vectors `(p1, p2)` and `(p2, p3)` using dot product.
- Average the cosine similarity of these angles, threshold at 0.5 to return 1.0 or 0.0.
- **Example:** Smooth trajectory with consistent angles: Score = 1.0.

#### `circulation_completion`
- **Process:** Sum angle changes to measure total circulation.
- Compute angle differences between consecutive points relative to the center.
- Sum absolute differences, divide by 360, clip to `[0, 1]`.
- **Example:** Trajectory completing 270°: Score = 0.75.

#### `path_efficiency`
- **Process:** Ratio of straight-line distance to total distance.
- Straight-line distance: `sqrt((end_x - start_x)^2 + (end_y - start_y)^2)`.
- Total distance: Sum of distances between consecutive points.
- Compute `efficiency = straight_line / total_distance`, threshold at 0.7.
- **Example:** Straight path: Efficiency = 1.0.

#### `directional_variance`
- **Process:** Calculate variation in movement direction.
- Compute direction angles between consecutive points using `atan2`.
- Calculate variance of these angles.
- **Example:** Erratic trajectory with high angle variance: High score.

### Summary
These features are derived systematically from raw coordinates to capture geometric and behavioral violations, providing a robust foundation for future classification using DBSCAN and RandomForest.

# Detailed Explanation of Algorithms Mentioned in the Document

## Algorithms Proposed for the Project

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

#### Definition
DBSCAN is an unsupervised clustering algorithm that groups data points based on their density, identifying clusters as dense regions separated by sparse areas and labeling sparse points as outliers.

#### Working Mechanism
1. **Initialization:** Start with an unvisited point and mark it as visited.
2. **Density Check:** If the point has at least `min_samples` neighbors within a distance `eps`, form a cluster.
3. **Cluster Expansion:** Recursively add all density-reachable points (points within `eps` of any point in the cluster) to the cluster.
4. **Outlier Identification:** Points not part of any cluster are labeled as noise.
- **Key Parameters:**
  - `eps`: Maximum distance between two points to be considered part of the same cluster.
  - `min_samples`: Minimum number of points to form a dense region.

#### Strengths
- Does not require the number of clusters to be predefined.
- Handles arbitrary-shaped clusters and outliers effectively.
- Robust to noise in spatial data.

#### Limitations
- Sensitive to the choice of `eps` and `min_samples`, requiring tuning.
- May struggle with varying density clusters or high-dimensional data without preprocessing.

#### Relevance to Your Project
DBSCAN is used in the `derive_roundabout_geometry` step to cluster normal trajectory points, filtering outliers to estimate the roundabout’s center and radius. Its ability to handle non-spherical, circular patterns and detect outliers makes it ideal for identifying deviations in trajectory data.

---

### RandomForest

#### Definition
RandomForest is a supervised ensemble learning algorithm that constructs multiple decision trees and aggregates their predictions to improve accuracy and reduce overfitting.

#### Working Mechanism
1. **Bootstrap Sampling:** Randomly sample the dataset with replacement to create multiple subsets.
2. **Feature Randomness:** At each node of a decision tree, select a random subset of features to determine the best split, reducing correlation between trees.
3. **Tree Construction:** Build independent decision trees on each subset.
4. **Aggregation:** For classification, take a majority vote across all trees to predict the class.
- **Key Parameters:**
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of each tree.
  - `class_weight`: Weights for imbalanced classes.

#### Strengths
- Handles high-dimensional data and noisy inputs effectively.
- Provides feature importance scores for interpretability.
- Reduces overfitting through averaging.

#### Limitations
- Can be computationally intensive with a large number of trees.
- Less interpretable at the individual tree level compared to a single decision tree.

#### Relevance to Your Project
RandomForest will be used as the classifier to process derived features (e.g., `central_island_violation`, `path_efficiency`). Its ability to handle class imbalance (375 normal vs. 185 abnormal trajectories) and provide feature importance aligns with the goal of robust and interpretable classification.

---

## Algorithms from the Literature Survey

### Zone-based Detection with Image Processing and Rule-based Classification (Bhavsar et al., 2023)

#### Definition
This is a hybrid methodology combining image processing techniques with rule-based classification, tailored for traffic violation detection using UAV video data.

#### Working Mechanism
1. **Image Processing:** Extract vehicle positions from UAV video frames using object detection (e.g., background subtraction or deep learning models like YOLO).
2. **Zone Assignment:** Divide the roundabout into predefined zones based on spatial coordinates and assign vehicle positions to these zones.
3. **Rule-based Classification:** Apply predefined rules (e.g., entering an exit zone without circulation) to classify trajectories as violating or non-violating.
- **Key Parameters:** Zone boundaries, rule thresholds (e.g., minimum time in a zone).

#### Strengths
- Intuitive for spatial violation detection in roundabouts.
- Relies on visual data, aligning with real-world traffic monitoring.

#### Limitations
- Dependent on the accuracy of image processing and zone definition.
- Less flexible for complex or dynamic traffic patterns.

#### Relevance to Your Project
This method inspires our `assign_zone_and_lane` function, where we use spatial coordinates to assign zones, though our data comes from CSV files rather than raw video, making our approach more data-driven.

---

### Statistical Analysis and Clustering (e.g., k-Means, Hierarchical Clustering) (Tselentis & Papadimitriou, 2023)

#### Definition
Statistical analysis involves descriptive and inferential methods to summarize data, often paired with clustering (e.g., k-Means or hierarchical clustering) to group similar patterns.

#### Working Mechanism
- **Statistical Analysis:** Compute metrics like mean, variance, or correlation to describe trajectory characteristics (e.g., speed, direction).
- **k-Means Clustering:**
  1. Initialize `k` cluster centroids randomly.
  2. Assign each point to the nearest centroid.
  3. Recalculate centroids as the mean of assigned points.
  4. Repeat until convergence.
  - **Key Parameters:** `k` (number of clusters).
- **Hierarchical Clustering:** Build a tree of clusters by iteratively merging or splitting based on distance metrics (e.g., Euclidean).
  - **Key Parameters:** Distance threshold, linkage method (e.g., single, complete).

#### Strengths
- k-Means is computationally efficient for spherical clusters.
- Hierarchical clustering provides a hierarchy of patterns, useful for exploratory analysis.

#### Limitations
- k-Means requires a predefined `k` and assumes spherical clusters, unsuitable for our circular trajectories.
- Hierarchical clustering is computationally expensive for large datasets.

#### Relevance to Your Project
These methods inform pattern recognition, but their limitations (e.g., needing `k`, handling outliers) make DBSCAN a better fit for our geometry derivation.

---

### Deep Learning-based Temporal Clustering with Convolutional Neural Networks (CNNs) and Explainability Techniques (Feng et al.)

#### Definition
This is a deep learning approach using CNNs to cluster time-series data (e.g., driving styles) with techniques to enhance model interpretability.

#### Working Mechanism
1. **Data Preprocessing:** Convert trajectory data into time-series format (e.g., speed over time).
2. **CNN Training:** Use CNNs to extract spatial-temporal features from the time-series, followed by a clustering layer (e.g., k-Means on learned features).
3. **Explainability:** Apply techniques like SHAP or LIME to interpret feature contributions to clusters.
- **Key Parameters:** Number of convolutional layers, filter sizes, learning rate.

#### Strengths
- Captures complex temporal patterns effectively.
- Provides explainability, enhancing trust in results.

#### Limitations
- Requires large datasets and significant computational resources.
- Less suitable for spatial data or small datasets like ours (560 trajectories).

#### Relevance to Your Project
While insightful for temporal clustering, its resource demands and focus on time-series data make it less applicable than DBSCAN for our spatial feature derivation.

---

### Spatial-Temporal Modeling with Kalman Filtering and Pattern Recognition Techniques (Wu et al.)

#### Definition
This combines Kalman filtering (a recursive algorithm for state estimation) with pattern recognition to model and classify trajectories.

#### Working Mechanism
- **Kalman Filtering:**
  1. Predict the next state (e.g., vehicle position) based on a motion model.
  2. Update the prediction using new measurements (e.g., video frames).
  - **Key Parameters:** Process noise, measurement noise.
- **Pattern Recognition:** Identify recurring spatial-temporal patterns (e.g., straight paths) using statistical or machine learning techniques.
- **Output:** Classified trajectories based on pattern matches.

#### Strengths
- Effective for tracking and predicting vehicle motion.
- Handles noisy data well.

#### Limitations
- Assumes linear motion models, which may not fit roundabout curves.
- Computationally intensive for real-time applications.

#### Relevance to Your Project
Kalman filtering could enhance trajectory smoothing, but its linear focus makes it less suitable than DBSCAN for our non-linear roundabout geometry.

---

### Review of Algorithms (e.g., k-Nearest Neighbors, Support Vector Machines, Clustering Methods) (Bian et al.)

#### Definition
This is a survey of various classification and clustering algorithms applied to trajectory data.

#### Working Mechanism
- **k-Nearest Neighbors (k-NN):**
  - Assign a point to the class of the majority of its `k` nearest neighbors based on distance (e.g., Euclidean).
  - **Key Parameters:** `k` (number of neighbors).
- **Support Vector Machines (SVM):**
  - Find the optimal hyperplane to separate classes, using kernel tricks (e.g., RBF) for non-linear data.
  - **Key Parameters:** Kernel type, regularization parameter (C).
- **Clustering Methods:** Includes k-Means, DBSCAN, etc., as described above.

#### Strengths
- k-NN is simple and effective for small datasets.
- SVM handles high-dimensional data with good generalization.
- Clustering methods (e.g., DBSCAN) suit unsupervised tasks.

#### Limitations
- k-NN is computationally expensive for large datasets and sensitive to distance metrics.
- SVM requires careful kernel selection and can be slow with many features.
- Clustering methods vary in suitability based on data shape and prior knowledge.

#### Relevance to Your Project
This review guides our algorithm choice, validating DBSCAN for clustering and RandomForest for classification due to their fit with our spatial data and imbalance.

---

### Summary
- **DBSCAN and RandomForest** are chosen for their complementary strengths in unsupervised clustering (geometry derivation) and supervised classification (future use), respectively.
- **Literature Algorithms** (e.g., k-Shape, k-Means, DTW, Deep Clustering, Kalman Filtering) offer insights but are less suitable due to temporal focus, cluster shape assumptions, or resource demands, reinforcing our methodological direction.

# Detailed Explanation: Path Efficiency in Vehicle Trajectory Classification

## What is Path Efficiency?

Path efficiency is a metric used to quantify how directly a vehicle travels from its starting point to its endpoint relative to the total distance it covers along its trajectory. In the context of your project, it measures the efficiency of a vehicle's path through a roundabout by comparing the straight-line (Euclidean) distance between the start and end points to the actual path length traveled. It is expressed as a ratio:

\[
\text{Path Efficiency} = \frac{\text{Straight-Line Distance}}{\text{Total Path Distance}}
\]

- **Range:** The value ranges from 0 to 1, where:
  - 1 indicates a perfectly straight path (maximum efficiency).
  - Values closer to 0 indicate a highly circuitous or inefficient path (e.g., due to detours or loops).
- **Interpretation:** In a roundabout, a high path efficiency might suggest a vehicle took a direct route (e.g., entering and exiting without significant circulation), while a low value might indicate normal circulation or abnormal behavior like U-turns or central island crossings.

## Why Are We Deriving Path Efficiency?

### Importance in Trajectory Classification
- **Distinguishing Normal vs. Abnormal Behavior:** Roundabouts enforce specific circulation rules (e.g., counterclockwise flow), and abnormal trajectories (e.g., cutting through the central island or making U-turns) often result in inefficient paths due to unnecessary detours or shortcuts. Deriving path efficiency helps identify these deviations, as abnormal trajectories typically have lower efficiency compared to the expected circular or semi-circular paths of normal trajectories.
- **Geometric Insight:** Path efficiency provides a geometric perspective on how vehicles navigate the roundabout, complementing other features like `central_island_violation` or `curvature_adherence`. It captures the overall navigational strategy, which is critical for understanding rule adherence.
- **Feature Diversity:** Including path efficiency adds a global measure of trajectory behavior, enhancing the feature set’s ability to differentiate between classes. This aligns with the goal of creating a robust dataset for future classification using DBSCAN and RandomForest.

### Relevance to Safety and Traffic Management
- **Safety Implications:** Inefficient paths (e.g., U-turns or shortcuts) can indicate risky maneuvers that increase collision risks at roundabouts. By deriving this feature, we contribute to traffic safety assessments, a key objective in your research.
- **Traffic Flow Analysis:** Understanding path efficiency helps assess how well vehicles follow designed traffic patterns, aiding urban planners in optimizing roundabout designs.

## How Are We Deriving Path Efficiency?

### Derivation Process
The derivation of path efficiency involves the following steps, starting from the raw trajectory data:

1. **Data Input:**
   - Raw data is stored in CSV files with columns `frameNo`, `left`, `top`, `w`, and `h`.
   - Each row represents a vehicle’s position at a specific frame, with `x_center = left + w/2` and `y_center = top + h/2` calculated during preprocessing using the `df_to_points` function.

2. **Extract Start and End Points:**
   - Identify the first point `(x_start, y_start)` and the last point `(x_end, y_end)` from the list of `(frameNo, x_center, y_center)` tuples for a given trajectory.

3. **Calculate Straight-Line Distance:**
   - Compute the Euclidean distance between the start and end points using the formula:
     \[
     \text{Straight-Line Distance} = \sqrt{(x_{\text{end}} - x_{\text{start}})^2 + (y_{\text{end}} - y_{\text{start}})^2}
     \]

4. **Calculate Total Path Distance:**
   - Sum the Euclidean distances between consecutive points along the trajectory. For a list of points `[(frame_1, x_1, y_1), (frame_2, x_2, y_2), ..., (frame_n, x_n, y_n)]`, the total distance is:
     \[
     \text{Total Path Distance} = \sum_{i=1}^{n-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}
     \]

5. **Compute Path Efficiency:**
   - Divide the straight-line distance by the total path distance:
     \[
     \text{Path Efficiency} = \frac{\text{Straight-Line Distance}}{\text{Total Path Distance}}
     \]
   - Apply a threshold (e.g., 0.7) to categorize efficiency, where values above the threshold might indicate efficient navigation, and values below might suggest detours or violations.

6. **Example:**
   - Trajectory: `[(1, 100.0, 200.0), (2, 102.0, 198.0), (3, 105.0, 195.0)]`
   - Start: `(100.0, 200.0)`, End: `(105.0, 195.0)`
   - Straight-Line Distance: \(\sqrt{(105.0 - 100.0)^2 + (195.0 - 200.0)^2} = \sqrt{25 + 25} = 7.07\)
   - Total Path Distance: \(\sqrt{(102.0 - 100.0)^2 + (198.0 - 200.0)^2} + \sqrt{(105.0 - 102.0)^2 + (195.0 - 198.0)^2} = 2.83 + 4.24 = 7.07\)
   - Path Efficiency: \(7.07 / 7.07 = 1.0\) (perfectly efficient, straight path)

## Why Are We Deriving It This Way?

### Methodological Rationale
- **Geometric Basis:** The use of Euclidean distances leverages the spatial nature of trajectory data, aligning with the roundabout’s circular geometry derived via `derive_roundabout_geometry`. This ensures the feature reflects the physical layout of the roundabout.
- **Simplicity and Interpretability:** The ratio-based approach is straightforward to compute and interpret, making it a valuable addition to our feature set for future classification with RandomForest, which benefits from interpretable inputs.
- **Scalability:** This method can be applied across all 560 trajectories in our dataset, accommodating the variability in directories (10, 11, 12) and the 2:1 class imbalance (375 normal, 185 abnormal).

### Alignment with Research Goals
- **Violation Detection:** By capturing deviations from efficient paths, we can flag potential abnormal behaviors (e.g., U-turns lowering efficiency) without relying solely on zone-specific features.
- **Feature Complementarity:** It complements features like `curvature_adherence` (local smoothness) and `circulation_completion` (total angle), providing a holistic view of trajectory behavior.

## Potential Noise and Bias Concerns

### Will Path Efficiency Be Noisy?
- **Noise Sources:** Yes, path efficiency can be noisy due to:
  - **Data Interpolation Errors:** The interpolated video data may introduce slight inaccuracies in `x_center` and `y_center`, affecting distance calculations.
  - **Tracking Errors:** Misalignment in bounding box coordinates (e.g., due to occlusion or lighting) can lead to erratic point sequences, inflating or deflating the total path distance.
  - **Frame Rate Variability:** Inconsistent frame intervals might skew the total distance if points are not evenly spaced.

- **Mitigation:** Preprocessing steps like filtering invalid points (e.g., NaN values) and normalizing distances can reduce noise. Additionally, aggregating efficiency over multiple frames can smooth out minor fluctuations.

### Does It Favor Straight Normal Paths but Consider Normal Turns as Abnormal?
- **Bias Toward Straight Paths:** Yes, path efficiency inherently favors straight-line travel because the straight-line distance is the shortest possible path. In a roundabout, normal trajectories typically involve turns (e.g., a 90° or 180° circulation), resulting in a total path distance greater than the straight-line distance, thus lowering the efficiency score (e.g., 0.5-0.7 for a quarter-circle turn).
- **Impact on Normal Turns:** This could lead to misclassification of normal turns as abnormal if the threshold is set too high (e.g., >0.7). For instance:
  - A normal 90° turn might have a straight-line distance of 100 units and a total path distance of 141 units (due to the hypotenuse of a right triangle), yielding an efficiency of \(100 / 141 \approx 0.71\).
  - If the threshold is 0.7, this normal turn would be borderline or classified as inefficient/abnormal.
- **Why This Happens:** Roundabouts are designed for circulation, not straight travel, so efficient paths (high efficiency) are less common unless a vehicle enters and exits at the same point (e.g., a U-turn, which is abnormal).

### Addressing the Bias
- **Contextual Thresholding:** Adjust the efficiency threshold based on expected roundabout behavior. For example, set it to 0.5 or lower, reflecting that normal circulation involves turns, while values near 0 (e.g., due to U-turns or island crossings) indicate abnormalities.
- **Combination with Other Features:** Use `curvature_adherence` and `circulation_completion` to contextualize efficiency. A low efficiency with high curvature adherence (smooth turns) suggests a normal circulation, while low efficiency with low adherence (erratic movements) indicates an abnormal path.
- **Normalization:** Normalize efficiency by the minimum expected path length for a given start-end pair (e.g., based on roundabout radius), reducing bias toward straight paths.

### Conclusion on Noise and Bias
- **Noise Management:** While noise is a concern, preprocessing and aggregation can mitigate its impact, ensuring path efficiency remains a reliable feature.
- **Bias Mitigation:** The potential to misclassify normal turns as abnormal can be addressed with a lower threshold and multi-feature analysis, aligning path efficiency with roundabout-specific expectations rather than assuming straight paths are the norm.
