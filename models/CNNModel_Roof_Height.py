import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# Load the data
data_path = '../data/Vienna/modelling_results/building_3d_Roofs_2.csv'
images_folder = '../data/Vienna/modelling_results/masked_orthophoto'
df = pd.read_csv(data_path)

df = df[df['roof_height'] <= 15]

# Prepare the data
image_size = (128, 128)  # Resize images to a fixed size

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1]
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

# Apply log transformation to the heights
#heights_log = np.log1p(heights)  # log1p is used to handle log(0)

# Check if there are any images and heights loaded
if len(images) == 0 or len(heights) == 0:
    raise ValueError("No images or heights were loaded. Check the images folder and file paths.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, heights, test_size=0.2, random_state=42)

# Plot the distribution of y_train_log and y_test_log
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(y_train, bins=30, edgecolor='black')
plt.title('Distribution of y_train (log-transformed)')
plt.xlabel('Log-transformed Roof Height')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(y_test, bins=30, edgecolor='black')
plt.title('Distribution of y_test (log-transformed)')
plt.xlabel('Log-transformed Roof Height')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Add dropout layer
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),  # Add L2 regularization
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Add dropout layer
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),  # Add L2 regularization
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # Add dropout layer
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),  # Add L2 regularization
    Dropout(0.5),  # Add dropout layer
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test Loss (RMSE): {sqrt(test_loss)}")

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the mean absolute error in the log-transformed scale
mae = mean_absolute_error(y_test, y_pred)
print(f"Test Loss (MAE): {mae}")

# Save the model
model.save('roof_height_prediction_model.h5')

# Plot the training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Roof Heights')
plt.ylabel('Predicted Roof Heights')
plt.title('Actual vs Predicted Roof Heights')
plt.show()
