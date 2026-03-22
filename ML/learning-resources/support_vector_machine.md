# Support Vector Machine (SVM)

Table of Contents
- [1. Support Vector Regression (SVR)](#1-support-vector-regression-svr)
- [2. Kernel Functions](#2-kernel-functions)

Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used for classification and regression tasks. It works by finding the optimal hyperplane that best separates the data points of different classes in a high-dimensional space. SVM can handle both linear and non-linear data by using kernel functions, making it versatile for various applications.

The main idea behind SVM is to maximize the margin between the closest data points of different classes, known as support vectors. The optimal hyperplane is the one that maximizes this margin, which helps in improving the generalization of the model on unseen data.


<b>Main Usage</b>

1. Classification: SVM is commonly used for binary and multi-class classification problems. It can effectively separate classes in high-dimensional spaces, making it suitable for tasks like image recognition, text classification, and bioinformatics.
2. Regression: SVM can also be used for regression tasks, known as Support Vector Regression (SVR). It aims to find a function that approximates the relationship between the independent and dependent variables while maximizing the margin of tolerance.

<b>Example in plain english</b>

Suppose we have a dataset of two types of fruits, apples and oranges, with features such as weight and color. We want to classify these fruits based on their features. SVM will find the best hyperplane that separates the apples from the oranges in the feature space. 

If the data is not linearly separable, SVM can use kernel functions to transform the data into a higher-dimensional space where it can find a linear separation.

Formula for SVM:

$$f(x) = w \cdot x + b$$

Where:
- f(x) is the output of the SVM model
- w is the weight vector
- x is the input feature vector
- b is the bias term

The SVM algorithm can be implemented using libraries such as scikit-learn in Python, which provides a simple interface for training and evaluating SVM models.

## 1. Support Vector Regression (SVR)

Support Vector Regression (SVR) is a variant of SVM used for regression tasks. It aims to find a function that approximates the relationship between the independent and dependent variables while maximizing the margin of tolerance. SVR uses a similar approach to SVM but focuses on fitting a curve to the data rather than finding a hyperplane for classification.

1. The Core Concept: The $\epsilon$-TubeIn SVR, we define a Tube around our regression line with a width of $\epsilon$ (epsilon).Inside the tube: Errors are ignored. We consider these points "correct enough."Outside the tube: These points are "Support Vectors." The model penalizes their distance from the tube.

2. The Mathematical FormulaThe goal of SVR is to find a function $f(x)$ that has at most $\epsilon$ deviation from the actual targets $y$ for all the training data, and at the same time is as flat as possible.The linear formula looks like this:$$y = \sum_{i=1}^{N} (\alpha_i - \alpha_i^*) K(x_i, x) + b$$

Breaking down the components:

$(\alpha_i - \alpha_i^*)$: These are the Lagrange Multipliers. Most of these will be zero! Only the points sitting on or outside the tube (the Support Vectors) have non-zero values. This makes SVR "sparse" and very efficient.

$K(x_i, x)$: This is the Kernel Function. This is the "magic" of SVR. It allows the model to handle non-linear data by mapping it into a higher-dimensional space.

$b$: The bias (intercept).

## 2. Kernel Functions

Kernel functions are used in SVM to transform the input data into a higher-dimensional space where it can be linearly separated. Common kernel functions include:
- Linear Kernel: $$K(x_i, x_j) = x_i \cdot x_j$$
- Polynomial Kernel: $$K(x_i, x_j) = (x_i \cdot x_j + 1)^d$$
- Radial Basis Function (RBF) Kernel: $$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$
- Sigmoid Kernel: $$K(x_i, x_j) = \tanh(\alpha x_i \cdot x_j + c)$$

Which Kernel to use depends on the specific problem and the nature of the data. The RBF kernel is often a good default choice for non-linear problems, while the linear kernel can be effective for linearly separable data. It is important to experiment with different kernels and parameters to find the best fit for your dataset.

## 3. Feature Scaling

When applying <b>Feature Scaling</b> to SVR, it is crucial to scale both the input features and the target variable. This is because SVR is sensitive to the scale of the data, and unscaled data can lead to poor performance.

1. Scaling Input Features: The input features should be scaled to ensure that they contribute equally to the model. Common scaling techniques include Standardization (z-score normalization) and Min-Max Scaling.

2. Scaling Target Variable: The target variable should also be scaled to ensure that the SVR model can learn effectively. This is especially important if the target variable has a different scale than the input features.

## 4. Inverse Transforming the Predicted Output

After making predictions with the SVR model, the predicted output will be in the scaled form. To get the actual predicted value in the original scale, you need to inverse transform the predicted output using the same scaling parameters that were used to scale the target variable during training.
```python
y_pred = regressor.predict(sc_x.transform([[6.5]]))

print(sc_y.inverse_transform(y_pred.reshape(-1, 1)))
```