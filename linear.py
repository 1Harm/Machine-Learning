import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Feature (independent variable)
y = np.array([2, 4, 5, 4, 5])  # Target (dependent variable)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on new data points
X_new = np.array([6, 7, 8]).reshape(-1, 1)
y_pred = model.predict(X_new)

# Plot the data points and the regression line
plt.scatter(X, y, label='Data Points')
plt.plot(X_new, y_pred, color='red', label='Linear Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
