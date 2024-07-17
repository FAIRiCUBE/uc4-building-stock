import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from math import sqrt

# Load the data
data_path = '../data/Vienna/modelling_results/building_3d_Roofs_2.csv'
df = pd.read_csv(data_path)

# Filter out outliers (roof heights of more than 15 meters)
df = df[(df['roof_height'] >= 1) & (df['roof_height'] <= 25)]

# Get the heights
heights = df['roof_height'].values

# Check if there are any heights loaded
if len(heights) == 0:
    raise ValueError("No heights were loaded. Check the data file.")

# Split data into training and testing sets
y_train, y_test = train_test_split(heights, test_size=0.2, random_state=42)

# Dummy model predictions (always predict the mean of the training set)
mean_train = np.mean(y_train)
y_pred = np.full_like(y_test, mean_train)

# Calculate and print the Mean Squared Error (MSE)
test_loss = mean_squared_error(y_test, y_pred)
print(f"Test Loss (MSE): {test_loss}")

# Calculate and print the Root Mean Squared Error (RMSE)
rmse = sqrt(test_loss)
print(f"Test Loss (RMSE): {rmse}")

# Calculate and print the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test Loss (MAE): {mae}")
