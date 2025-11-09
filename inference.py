"""
Dogs vs Cats Classifier - Local Inference Script
Downloads the model from Hugging Face Hub and runs predictions on uploaded images.

Author: Abdul Ahad (990aa)
"""

import os
import sys
import numpy as np
import joblib
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "a-01a/dogs-vs-cats-svm"
MODEL_FILENAME = "svm_vgg16_cats_dogs.pkl"
IMAGE_SIZE = (224, 224)


def download_model():
    """Download model from Hugging Face Hub"""
    print("📥 Downloading model from Hugging Face Hub...")
    try:
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
        print(f"✓ Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None


def load_model(model_path):
    """Load the trained model and feature extractor"""
    print("🔧 Loading model components...")
    try:
        # Load VGG16 for feature extraction (without top layers)
        vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        print("✓ VGG16 feature extractor loaded")
        
        # Load the trained SVM model
        model_data = joblib.load(model_path)
        svm_model = model_data['model']
        pca = model_data['pca']
        print("✓ SVM model and PCA loaded")
        
        return vgg_model, svm_model, pca
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None


def preprocess_image(img_path):
    """Preprocess image for prediction"""
    try:
        # Load and resize image
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None


def predict_image(img_path, vgg_model, svm_model, pca):
    """Predict whether the image is a cat or dog"""
    # Preprocess image
    img_array = preprocess_image(img_path)
    if img_array is None:
        return None, None
    
    # Extract features using VGG16
    features = vgg_model.predict(img_array, verbose=0)
    features_flat = features.reshape(1, -1)
    
    # Apply PCA
    features_pca = pca.transform(features_flat)
    
    # Predict with SVM
    prediction = svm_model.predict(features_pca)[0]
    
    # Get confidence score
    try:
        decision_score = svm_model.decision_function(features_pca)[0]
        # Convert to probability score
        confidence = 1 / (1 + np.exp(-decision_score))
        if prediction == 0:  # Cat
            confidence = 1 - confidence
    except:
        confidence = 0.5
    
    return prediction, confidence


def main():
    """Main execution function"""
    print("=" * 70)
    print("🐱🐶 Dogs vs Cats Classifier - Local Inference")
    print("=" * 70)
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("\n❌ Usage: python inference.py <path_to_image>")
        print("\nExample:")
        print("   python inference.py my_pet.jpg")
        print("   python inference.py C:\\Users\\username\\Downloads\\dog.png")
        return
    
    img_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"\n❌ Image not found: {img_path}")
        return
    
    print(f"\n📸 Image: {img_path}")
    
    # Download model from Hugging Face
    model_path = download_model()
    if model_path is None:
        return
    
    # Load model components
    vgg_model, svm_model, pca = load_model(model_path)
    if vgg_model is None or svm_model is None or pca is None:
        return
    
    # Make prediction
    print("\n🔮 Making prediction...")
    prediction, confidence = predict_image(img_path, vgg_model, svm_model, pca)
    
    if prediction is None:
        print("❌ Prediction failed")
        return
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 PREDICTION RESULTS")
    print("=" * 70)
    
    label = "🐱 Cat" if prediction == 0 else "🐶 Dog"
    print(f"\n🎯 Prediction: {label}")
    print(f"📈 Confidence: {confidence:.2%}")
    
    if confidence > 0.8:
        print(f"✅ High confidence - Very likely a {label}")
    elif confidence > 0.6:
        print(f"⚠️  Medium confidence - Probably a {label}")
    else:
        print("❓ Low confidence - Uncertain prediction")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
