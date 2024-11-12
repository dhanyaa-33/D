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
