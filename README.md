# Identifying Abnormal Driving Behavior Using Spatio-Temporal Analysis  

## Weekly Progress Update  

This week, we focused on exploring different machine learning algorithms to identify abnormal driving behavior using spatio-temporal data (location and time). Since driving patterns change over time, we looked into time-series clustering methods that can help group similar driving behaviors without needing labeled data.  

## Algorithms Explored  

### K-Shape Clustering  
K-Shape Clustering groups driving behaviors based on the overall shape of their patterns over time. Unlike traditional clustering methods, K-Shape uses Shape-Based Distance (SBD) to compare time-series data. This makes it more effective for identifying different driving styles, such as aggressive or distracted driving. Research has shown that K-Shape is highly accurate and efficient for time-series clustering, making it a strong candidate for our project.  

Paper References:  
- [ACM Research on K-Shape](https://dl.acm.org/doi/pdf/10.1145/2723372.2737793)  
- [IEEE Research on K-Shape](https://ieeexplore.ieee.org/abstract/document/10192069/references#references)  

### Dynamic Time Warping (DTW) with K-Means or Hierarchical Clustering  
DTW is useful when comparing driving patterns that do not happen at the same speed or time. For example, two drivers might follow the same acceleration pattern, but one does it faster than the other. DTW helps align these time shifts, making it a good method for detecting subtle changes in driving behavior.  

Paper Reference:  
- [IEEE Paper on DTW-Based Clustering](https://ieeexplore.ieee.org/document/10057714)  

### Time-Series K-Medoids  
K-Medoids is similar to K-Shape but more robust to outliers. Instead of calculating an "average" pattern for each cluster, K-Medoids picks actual examples from the dataset as cluster centers. This makes it especially useful for real-world driving data, which often contains unexpected variations.  

Paper References:  
- [IEEE Research on K-Medoids](https://ieeexplore.ieee.org/document/9888389/keywords#keywords)  
- [Springer Chapter on Time-Series Clustering](https://link.springer.com/chapter/10.1007/978-981-99-3284-9_5)  

## Next Steps  
- We have decided to work further on K-Shape Clustering for detecting abnormal driving behavior.  
- The next step is to implement and test this algorithm on real-world driving datasets.  
- We will also compare K-Shape with DTW and K-Medoids to evaluate performance.  


