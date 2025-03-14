import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib  # For saving the XGBoost model
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# Load the data
data_path = 'buildings_rooftops.csv'
images_folder = 'masked_orthophoto'
df = pd.read_csv(data_path)

# Filter out outliers (roof heights of more than 8 meters)
df = df[(df['roof_height'] <= 8)]

# Function to categorize roof heights into classes
def categorize_height(height):
    if height <= 1.789:
        return 0  # class1
    elif height <= 3.37:
        return 1  # class2
    elif height <= 4.591:
        return 2  # class3
    else:
        return 3  # class4
    # if height <= 3.37:
    #     return 0  
    # else:
    #     return 1  

# Apply the function to create the target variable
df['roof_height_class'] = df['roof_height'].apply(categorize_height)

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
height_classes = []
for idx, row in df.iterrows():
    object_id = row['object_id']
    roof_height_class = row['roof_height_class']
    image_path = os.path.join(images_folder, f"{object_id}.jpg")
    if os.path.exists(image_path):
        images.append(load_and_preprocess_image(image_path))
        height_classes.append(roof_height_class)

# Convert to numpy arrays
images = np.array(images)
height_classes = np.array(height_classes)

# Check if there are any images and heights loaded
if len(images) == 0 or len(height_classes) == 0:
    raise ValueError("No images or height classes were loaded. Check the images folder and file paths.")

print(len(images))
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, height_classes, test_size=0.2, random_state=42)

# Unfreeze the last few layers for fine-tuning
# for layer in base_model.layers[-10:]:
#     layer.trainable = True

# Load the pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
#base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

# Extract features from images using VGG16
X_train_features = base_model.predict(X_train)
X_test_features = base_model.predict(X_test)

# Flatten the features to make them suitable for XGBoost
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# Train the XGBoost classifier
# xgb_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
# xgb_classifier.fit(X_train_features, y_train)
# joblib.dump(xgb_classifier, 'xgb_classifier_Class.pkl')
# y_pred = xgb_classifier.predict(X_test_features)



# # Random Forest Classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train_features, y_train)
# joblib.dump(rf_classifier, 'RandomForrest_Class.pkl')
# y_pred = rf_classifier.predict(X_test_features)

# LightGBM Classifier
lgbm_classifier = LGBMClassifier(n_estimators=1000, learning_rate=0.1, max_depth=10, random_state=42)
lgbm_classifier.fit(X_train_features, y_train)
joblib.dump(lgbm_classifier, 'lgbm_classifier_Class.pkl')
y_pred = lgbm_classifier.predict(X_test_features)


# # Train the Gradient Boosting classifier
# gb_classifier = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42)
# gb_classifier.fit(X_train_features, y_train)
# joblib.dump(gb_classifier, 'gradient_boosting_classifier.pkl')
# y_pred = gb_classifier.predict(X_test_features)




# # Simple Neural Network Classifier
# nn_classifier = Sequential()
# nn_classifier.add(Dense(256, activation='relu', input_dim=X_train_features.shape[1]))
# nn_classifier.add(Dense(128, activation='relu'))
# nn_classifier.add(Dense(4, activation='softmax'))  # 4 classes
# nn_classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# nn_classifier.fit(X_train_features, y_train, epochs=20, batch_size=32, validation_split=0.1)
# joblib.dump(nn_classifier, 'nn_classifier_Class.pkl')
# y_pred = nn_classifier.predict(X_test_features)

# # After predicting the probabilities with the neural network
# y_pred_probs = nn_classifier.predict(X_test_features)

# # Convert probabilities to class labels by taking the argmax (index of the highest probability)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # Now you can compute accuracy and the confusion matrix
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Normalize the confusion matrix by dividing by the total number of samples
# cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# print("Normalized Confusion Matrix:")
# print(cm_normalized)

# # Plot normalized confusion matrix
# plt.figure(figsize=(6, 6))
# plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Normalized Confusion Matrix')
# plt.colorbar()
# plt.xticks(np.arange(2), ['class1', 'class2'], rotation=45)
# plt.yticks(np.arange(2), ['class1', 'class2'])
# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.savefig('Normalized_Confusion_Matrix_NN.pdf', format='pdf')


# Classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Normalize the confusion matrix by dividing by the total number of samples
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("Normalized Confusion Matrix:")
print(cm_normalized)

# Plot normalized confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(2), ['class1', 'class2'], rotation=45)
plt.yticks(np.arange(2), ['class1', 'class2'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('Normalized_Confusion_Matrix_lGB.pdf', format='pdf')