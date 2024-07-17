import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import joblib  # For saving the XGBoost model

# Load the data
data_path = '../data/Vienna/modelling_results/building_3d_Roofs_2.csv'
images_folder = '../data/Vienna/modelling_results/masked_orthophoto'
df = pd.read_csv(data_path)

# Filter out outliers (roof heights of more than 15 meters)
df = df[(df['roof_height'] >= 1) & (df['roof_height'] <= 25)]

# Prepare the data
image_size = (224, 224)  # Resize images to a fixed size for VGG16

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    return image

# Load images and their corresponding heights
images = []
heights = []
for idx, row in df.iterrows():
    object_id = row['object_id']
    roof_height = row['roof_height']
    image_path = os.path.join(images_folder, f"{object_id}.jpg")
    if os.path.exists(image_path):
        images.append(load_and_preprocess_image(image_path))
        heights.append(roof_height)

# Convert to numpy arrays
images = np.array(images)
heights = np.array(heights)

# Check if there are any images and heights loaded
if len(images) == 0 or len(heights) == 0:
    raise ValueError("No images or heights were loaded. Check the images folder and file paths.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, heights, test_size=0.2, random_state=42)

# Load the pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Extract features from images using VGG16
X_train_features = base_model.predict(X_train)
X_test_features = base_model.predict(X_test)

# Flatten the features to make them suitable for XGBoost
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# Train the XGBoost regressor
xgb_regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_regressor.fit(X_train_features, y_train)

# Save the XGBoost regressor model
joblib.dump(xgb_regressor, 'xgb_regressor.pkl')

# Make predictions
y_pred = xgb_regressor.predict(X_test_features)

# Calculate and print the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Test Loss (MSE): {mse}")

# Calculate and print the Root Mean Squared Error (RMSE)
rmse = sqrt(mse)
print(f"Test Loss (RMSE): {rmse}")

# Calculate and print the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test Loss (MAE): {mae}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Roof Heights')
plt.ylabel('Predicted Roof Heights')
plt.title('Actual vs Predicted Roof Heights')
plt.savefig('XGBoost_1_25.pdf', format='pdf')
plt.show()