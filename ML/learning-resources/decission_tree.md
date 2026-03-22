# Decision Tree (CART   Algorithm)

A Decision Tree is a supervised machine learning algorithm that can be used for both classification and regression tasks. It works by recursively splitting the data into subsets based on the feature that provides the best separation of the target variable. The resulting tree structure consists of nodes representing features, branches representing decision rules, and leaf nodes representing the final predictions.

The CART (Classification and Regression Trees) algorithm is a specific implementation of decision trees that can handle both classification and regression problems. It uses the Gini impurity or mean squared error as the criterion for splitting the data, depending on whether it is a classification or regression task.

What Happens when we fit a Decision Tree to the data?

When we fit a Decision Tree to the data, the algorithm starts at the root node and evaluates all the features to determine which one provides the best split based on the chosen criterion (Gini impurity for classification or mean squared error for regression). The data is then split into subsets based on the values of the selected feature. This process is repeated recursively for each subset, creating branches and nodes until a stopping criterion is met (e.g., maximum depth, minimum samples per leaf, or no further improvement in the criterion). The final result is a tree structure that can be used to make predictions by traversing the tree based on the feature values of the input data.

### Homework: 

Implement a Decision Tree from scratch in Python, without using any libraries. You can use the CART algorithm for both classification and regression tasks. Test your implementation on a simple dataset and compare the results with a Decision Tree implementation from a library like scikit-learn. Pokemon dataset can be a good choice for this task, as it has both classification and regression problems 

1. Classification: Predicting the type of a Pokemon based on its features (e.g., attack, defense, speed).
2. Regression: Predicting the total stats of a Pokemon based on its features.