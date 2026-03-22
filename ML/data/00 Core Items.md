## When to apply Feature Scaling?

Feature Scaling is a crucial preprocessing step in many machine learning algorithms, especially those that rely on the distance between data points, such as Support Vector Machines (SVM), K-Nearest Neighbors (KNN), and Principal Component Analysis (PCA). It ensures that all features contribute equally to the result and prevents features with larger ranges from dominating the learning process.

## When Not to Apply Feature Scaling?

1. Algorithms that are not sensitive to the scale of the features, such as tree-based algorithms (e.g., Decision Trees, Random Forests, Gradient Boosting) and Naive Bayes. These algorithms can handle features with different scales without any issues, as they are based on rules or probabilities rather than distance calculations.

2. When the features are already on the same scale, such as binary features (0 and 1) or categorical features that have been one-hot encoded. In these cases, feature scaling may not be necessary and could even distort the data.
