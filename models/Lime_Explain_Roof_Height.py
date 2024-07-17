import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import joblib  # For loading the XGBoost model

# Function to load and preprocess images
def load_and_preprocess_image(image_path, image_size):
    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    return image

# Load the pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Ensure the base model is not trainable

# Load the saved XGBoost model
xgb_regressor = joblib.load('xgb_regressor.pkl')

# Function to predict using the XGBoost model
def predict_fn(images):
    features = base_model.predict(images)
    features = features.reshape(features.shape[0], -1)
    return xgb_regressor.predict(features).reshape(-1, 1)

# Function to generate and plot LIME explanation for a given image
def plot_lime_explanation(image_path, model, base_model, image_size, save_path, actual_value):
    image_to_explain = load_and_preprocess_image(image_path, image_size)
    image_to_explain_features = base_model.predict(np.expand_dims(image_to_explain, axis=0))
    image_to_explain_features = image_to_explain_features.reshape(1, -1)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_to_explain.astype('double'), 
                                             predict_fn, 
                                             top_labels=1, 
                                             hide_color=0, 
                                             num_samples=1000)
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                 positive_only=False, 
                                                 num_features=10, 
                                                 hide_rest=False)
    
    original_image = load_img(image_path, target_size=image_size)
    original_image = img_to_array(original_image) / 255.0

    # Get the predicted value for the image
    predicted_value = model.predict(image_to_explain_features)[0]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(mark_boundaries(original_image, mask))
    plt.title(f"{os.path.basename(image_path)} - Real Value: {actual_value:.2f} - Predicted Value: {predicted_value:.2f}")
    plt.savefig(save_path, format='pdf')
    plt.show()

# Load the data
data_path = '../data/Vienna/modelling_results/building_3d_Roofs_2.csv'
images_folder = '../data/Vienna/modelling_results/masked_orthophoto'
df = pd.read_csv(data_path)

# Filter out outliers (roof heights of more than 15 meters)
df = df[(df['roof_height'] >= 1) & (df['roof_height'] <= 25)]

# Randomly select image paths from df['object_id']
num_samples = 3  # Number of images to randomly select
selected_objects = df['object_id'].sample(num_samples).tolist()

# Construct image paths and actual values
image_paths = [os.path.join(images_folder, f"{obj_id}.jpg") for obj_id in selected_objects]
actual_values = [df[df['object_id'] == obj_id]['roof_height'].values[0] for obj_id in selected_objects]

# Generate and save LIME explanations for the selected images
for image_path, actual_value in zip(image_paths, actual_values):
    if os.path.exists(image_path):  # Ensure the image file exists
        plot_lime_explanation(image_path, xgb_regressor, base_model, (224, 224), os.path.join("Explanations", f"LIME_Explanation_Overlay_{os.path.basename(image_path)}.pdf"), actual_value)
