"""
Dogs vs Cats Classifier
Loads the trained .keras model and makes predictions

Author: Abdul Ahad (990aa)
"""

import os
import sys
import numpy as np
import h5py
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

REPO_ID = "a-01a/dogs-vs-cats-svm"
FEATURE_EXTRACTOR_FILENAME = "cats-vs-dogs.keras"
COMPONENTS_FILENAME = "cats-vs-dogs-components.keras"
IMAGE_SIZE = (224, 224)
LOCAL_CACHE_DIR = "./model_cache"


def download_model_files():
    """Download .keras model files from Hugging Face Hub"""
    print("Downloading model from Hugging Face Hub...")
    try:
        os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
        
        feature_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FEATURE_EXTRACTOR_FILENAME,
            repo_type="model",
            cache_dir=LOCAL_CACHE_DIR
        )
        
        components_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=COMPONENTS_FILENAME,
            repo_type="model",
            cache_dir=LOCAL_CACHE_DIR
        )
        
        print("✓ Model files downloaded")
        return feature_path, components_path
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return None, None


def load_model_keras(feature_path, components_path):
    """Load model from .keras format"""
    print("Loading model components...")
    try:
        # Load feature extractor
        feature_extractor = load_model(feature_path)
        print("✓ Feature extractor loaded")
        
        # Load components from .keras file (HDF5)
        with h5py.File(components_path, 'r') as f:
            # Load PCA
            pca = PCA(n_components=f['pca'].attrs['n_components'])
            pca.components_ = f['pca/components'][:]
            pca.mean_ = f['pca/mean'][:]
            pca.explained_variance_ = f['pca/explained_variance'][:]
            pca.n_components_ = f['pca'].attrs['n_components']
            
            # Load Scaler
            scaler = StandardScaler()
            scaler.mean_ = f['scaler/mean'][:]
            scaler.scale_ = f['scaler/scale'][:]
            
            # Load SVM
            gamma_val = f['svm'].attrs['gamma']
            gamma = float(gamma_val) if isinstance(gamma_val, (int, float, np.number)) else gamma_val
            
            svm_model = svm.SVC(
                kernel=f['svm'].attrs['kernel'],
                C=f['svm'].attrs['C'],
                gamma=gamma,
                probability=True
            )
            
            svm_model.support_vectors_ = f['svm/support_vectors'][:]
            svm_model.dual_coef_ = f['svm/dual_coef'][:]
            svm_model._dual_coef_ = f['svm/dual_coef'][:]
            svm_model.intercept_ = f['svm/intercept'][:]
            svm_model._intercept_ = f['svm/intercept'][:]
            svm_model.support_ = f['svm/support'][:]
            svm_model._n_support = np.array([
                f['svm'].attrs['n_support_0'],
                f['svm'].attrs['n_support_1']
            ])
            svm_model._gamma = gamma
            svm_model.classes_ = np.array([0, 1])
            svm_model._sparse = False
            svm_model.shape_fit_ = (svm_model.support_vectors_.shape[0], svm_model.support_vectors_.shape[1])
            
            # Load probability calibration parameters if they exist
            if 'probA' in f['svm']:
                svm_model._probA = f['svm/probA'][:]
            else:
                svm_model._probA = np.array([])
                
            if 'probB' in f['svm']:
                svm_model._probB = f['svm/probB'][:]
            else:
                svm_model._probB = np.array([])
            
        print("✓ PCA, Scaler, and SVM loaded")
        return feature_extractor, svm_model, pca, scaler
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


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


def predict_image(img_path, feature_extractor, svm_model, pca, scaler):
    """Predict whether the image is a cat or dog"""
    img_array = preprocess_image(img_path)
    if img_array is None:
        return None, None
    
    # Extract features
    features = feature_extractor.predict(img_array, verbose=0)
    features_flat = features.reshape(1, -1)
    
    # Apply PCA
    features_pca = pca.transform(features_flat)
    
    # Apply scaling
    features_scaled = scaler.transform(features_pca)
    
    # Predict
    try:
        prediction = svm_model.predict(features_scaled)[0]
        probability = svm_model.predict_proba(features_scaled)[0]
        confidence = probability[prediction]
        
        return prediction, confidence
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None


def main():
    """Main execution function"""
    print("=" * 70)
    print("🐱🐶 Dogs vs Cats Classifier - Inference (.keras format)")
    print("=" * 70)
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("\n❌ Usage: python inference.py <path_to_image>")
        print("\nExample:")
        print("   python inference.py test/cat1.jpeg")
        return
    
    img_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"\n❌ Image not found: {img_path}")
        return
    
    print(f"\nImage: {img_path}")
    
    # Check if models exist locally first
    local_feature_path = "cats-vs-dogs.keras"
    local_components_path = "cats-vs-dogs-components.keras"
    
    if os.path.exists(local_feature_path) and os.path.exists(local_components_path):
        print("\n✓ Using local model files")
        feature_path = local_feature_path
        components_path = local_components_path
    else:
        # Download from HuggingFace
        print()
        feature_path, components_path = download_model_files()
        if feature_path is None:
            return
    
    # Load model components
    print()
    feature_extractor, svm_model, pca, scaler = load_model_keras(feature_path, components_path)
    if feature_extractor is None or svm_model is None or pca is None:
        return
    
    # Make prediction
    print("\nMaking prediction...")
    prediction, confidence = predict_image(img_path, feature_extractor, svm_model, pca, scaler)
    
    if prediction is None:
        print("❌ Prediction failed")
        return
    
    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    
    label = "🐱 Cat" if prediction == 0 else "🐶 Dog"
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    
    if confidence > 0.8:
        print(f"High confidence - Very likely a {label}")
    elif confidence > 0.6:
        print(f"Medium confidence - Probably a {label}")
    else:
        print("Low confidence - Uncertain prediction")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
