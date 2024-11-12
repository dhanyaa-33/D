from scipy.stats import skew

# Given data
data = [
    [65, 150, 5, 0, 1],
    [70, 160, 6, 1, 0],
    [75, 170, 4, 0, 1],
    [60, 140, 5, 1, 0],
    [80, 180, 3, 0, 1]
]

# Flatten the data into a single list
flattened_data = [item for sublist in data for item in sublist]

# Calculate skewness
skewness = skew(flattened_data)
print("Skewness of the data:", skewness)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset (replace with actual data if available)
data = {
    'SquareFootage': [1500, 2000, 1700, 2500, 1800, 1600],
    'Bedrooms': [3, 4, 3, 5, 3, 2],
    'Price': [300000, 400000, 350000, 500000, 360000, 320000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define features and target
X = df[['SquareFootage', 'Bedrooms']]
y = df['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Display model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


