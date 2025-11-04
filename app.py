import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

import gradio as gr
import numpy as np
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


# Load model artifacts
print("Loading model artifacts...")
try:
    model_artifacts = joblib.load('svm_vgg16_cats_dogs.pkl')
    svm_model = model_artifacts['svm_model']
    scaler = model_artifacts['scaler']
    pca = model_artifacts['pca']
    feature_extractor = model_artifacts['feature_extractor']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

IMG_WIDTH = 224
IMG_HEIGHT = 224


def predict_image(image):
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        features = feature_extractor.predict(img_array, verbose=0)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        prediction = svm_model.predict(features_pca)[0]
        probabilities = svm_model.predict_proba(features_pca)[0]
        
        result = {
            'Cat': float(probabilities[0]),
            'Dog': float(probabilities[1])
        }
        
        predicted_class = 'Cat' if prediction == 0 else 'Dog'
        confidence = float(probabilities[prediction])
        
        return result, f"**Prediction: {predicted_class}** (Confidence: {confidence:.2%})"
        
    except Exception as e:
        return {"Cat": 0.0, "Dog": 0.0}, f"Error: {str(e)}"


iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an image of a cat or dog"),
    outputs=[
        gr.Label(label="Classification Probabilities"),
        gr.Textbox(label="Prediction Result", lines=2)
    ],
    title="🐱🐶 Dogs vs Cats Classifier",
    description="SVM + VGG16 Transfer Learning Model",
    article="Trained using VGG16 feature extraction + SVM classification",
    theme="default",
    allow_flagging="never"
)


if __name__ == "__main__":
    iface.launch()
