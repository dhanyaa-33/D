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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Given data
data = {
    'Student_ID': [1, 2, 3, 4, 5, 6],
    'Math_Score': [85, 70, 90, 60, 95, 80],
    'Science_Score': [80, 65, 85, 55, 90, 75]
}

# Create DataFrame
df = pd.DataFrame(data)

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Math_Score', y='Science_Score')
plt.title('Math Score vs Science Score')
plt.show()

# Correlation matrix
correlation_matrix = df[['Math_Score', 'Science_Score']].corr()
print("Correlation Matrix:\n", correlation_matrix)

# Heatmap for correlation matrix
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

