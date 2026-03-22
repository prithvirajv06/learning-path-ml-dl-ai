# Polinomial Linear Regression

This regression technique is used when the relationship between the independent variable (X) and the dependent variable (y) is not linear but can be modeled as a polynomial. It allows us to capture more complex relationships between the variables. Exponential regression is a special case of polynomial regression where the degree of the polynomial is 1, but the independent variable is transformed using an exponential function.


Here the term linear refers to the fact that the coefficients of the polynomial are linear, even though the relationship between X and y is non-linear. The model can be expressed as: 
y = β0 + β1X + β2X^2 + ... + βnX^n + ε

# Linear Regression & Polynomial Features

The features are transformed into polynomial features, which allows the linear regression model to fit a non-linear relationship between the independent variable and the dependent variable. The degree of the polynomial can be adjusted to capture more complex relationships in the data. However, it is important to be cautious when increasing the degree of the polynomial, as it can lead to overfitting, where the model fits the training data too closely and performs poorly on unseen data.

Formula:
y = β0 + β1X + β2X^2 + ... + βnX^n + ε

### Degree of polynomial

The degree of the polynomial refers to the highest power of the independent variable (X) in the polynomial equation. For example, if the degree is 2, the model will include terms up to X^2. Increasing the degree allows the model to capture more complex relationships, but it also increases the risk of overfitting.

Example of polynomial regression with degree 2:
y = β0 + β1X + β2X^2 + ε

Example of polynomial regression with degree 4:
y = β0 + β1X + β2X^2 + β3X^3 + β4X^4 + ε

So it mean adding more features to the model, which can help capture more complex relationships in the data, but it also increases the risk of overfitting. It is important to choose the degree of the polynomial carefully based on the data and the problem at hand.

<b>Evaluation of the model</b> can be done using metrics such as R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). R-squared value indicates how well the model fits the data, with a value closer to 1 indicating a better fit. MAE, MSE, and RMSE measure the average error between the predicted values and the actual values, with lower values indicating better performance.

