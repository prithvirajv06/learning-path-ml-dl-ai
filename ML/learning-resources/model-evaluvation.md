# Model Evaluation

R Squared (R²) is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, where a value closer to 1 indicates a better fit of the model to the data. R² can be calculated using the formula:
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
Where:  
- $SS_{res}$ is the sum of squares of residuals (the difference between observed and predicted values).
- $SS_{tot}$ is the total sum of squares (the difference between observed values and the mean of observed values).

More the R² value, better the model fits the data. 

However, it is important to note that a high R² does not necessarily mean that the model is good, as it can be influenced by outliers and overfitting. It is always recommended to use R² in conjunction with other evaluation metrics and visualizations to assess the performance of a regression model.


### Adjusted R Squared (Adjusted R²) 

Adjusted R² is a modified version of R² that takes into account the number of independent variables in the model. It is used to prevent overfitting by penalizing the addition of irrelevant features. The formula for Adjusted R² is:

$$Adjusted R^2 = 1 - \left( \frac{SS_{res}/(n - k - 1)}{SS_{tot}/(n - 1)} \right)$$
Where:
- $SS_{res}$ is the sum of squares of residuals.
- $SS_{tot}$ is the total sum of squares.
- $n$ is the number of observations. Meaning the number of data points in the dataset. eg. if we have 100 rows in our dataset, then n = 100.
- $k$ is the number of independent variables. Meaning the number of features in our dataset. eg. if we have 5 columns in our dataset, then k = 5.

Point to note: 

1. Adjusted R² can be negative if the model is a poor fit for the data, which indicates that the model is worse than a simple horizontal line (mean of the dependent variable).

2. Adjusted R² will always be less than or equal to R², and it can decrease if irrelevant features are added to the model, which helps in identifying overfitting.

```python
from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R²:", r2)
```
