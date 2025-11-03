import joblib
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import sys
import os

# Configuration
MODEL_PATH = "svm_vgg16_cats_dogs.pkl"
IMG_WIDTH = 224
IMG_HEIGHT = 224

def load_model():
    """Load the trained model artifacts"""
    print("Loading model...")
    try:
        model_artifacts = joblib.load(MODEL_PATH)
        print("✓ Model loaded successfully!")
        return model_artifacts
    except FileNotFoundError:
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        print("Please ensure the model is trained and saved first.")
        sys.exit(1)


def predict_image(image_path, model_artifacts):
    """
    Predict whether an image contains a cat or dog
    
    Args:
        image_path: Path to the image file
        model_artifacts: Dictionary containing model components
        
    Returns:
        tuple: (prediction, confidence, probabilities)
    """
    # Extract components
    svm_model = model_artifacts['svm_model']
    scaler = model_artifacts['scaler']
    pca = model_artifacts['pca']
    feature_extractor = model_artifacts['feature_extractor']
    
    try:
        # Load and preprocess image
        print(f"\nProcessing image: {image_path}")
        img = Image.open(image_path)
        img = img.convert('RGB')  # Ensure RGB
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Convert to array and preprocess
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Extract features using VGG16
        print("Extracting features...")
        features = feature_extractor.predict(img_array, verbose=0)
        
        # Scale and apply PCA
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Make prediction
        print("Making prediction...")
        prediction = svm_model.predict(features_pca)[0]
        probabilities = svm_model.predict_proba(features_pca)[0]
        
        # Get results
        label = 'Cat' if prediction == 0 else 'Dog'
        confidence = probabilities[prediction]
        
        return label, confidence, probabilities
        
    except FileNotFoundError:
        print(f"❌ Error: Image file '{image_path}' not found!")
        return None, None, None
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None, None, None


def display_results(label, confidence, probabilities):
    """Display prediction results in a nice format"""
    if label is None:
        return
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n🎯 Prediction: {label}")
    print(f"📊 Confidence: {confidence:.2%}")
    print("\n📈 Class Probabilities:")
    print(f"   🐱 Cat: {probabilities[0]:.2%}")
    print(f"   🐶 Dog: {probabilities[1]:.2%}")
    print("\n" + "=" * 60)


def main():
    """Main execution function"""
    print("=" * 60)
    print("Dogs vs Cats Model - Local Test")
    print("=" * 60)
    
    # Load model
    model_artifacts = load_model()
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\nUsage: python test_model_locally.py <image_path>")
        print("\nExample:")
        print("  python test_model_locally.py cat.jpg")
        print("  python test_model_locally.py C:/path/to/dog.jpg")
        
        # Try to find a test image
        test_images = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if test_images:
            print(f"\nFound {len(test_images)} image(s) in current directory:")
            for img in test_images[:5]:  # Show max 5
                print(f"  - {img}")
            image_path = input("\nEnter image path (or press Enter to exit): ").strip()
            if not image_path:
                print("Exiting...")
                sys.exit(0)
        else:
            image_path = input("\nEnter image path: ").strip()
            if not image_path:
                print("No image path provided. Exiting...")
                sys.exit(0)
    
    # Make prediction
    label, confidence, probabilities = predict_image(image_path, model_artifacts)
    
    # Display results
    display_results(label, confidence, probabilities)
    
    # Offer to test another image
    print("\n Tip: You can also run:")
    print("   python test_model_locally.py path/to/your/image.jpg")
    print("\nModel test complete! ✓")


if __name__ == "__main__":
    main()
