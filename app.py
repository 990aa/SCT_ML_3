import gradio as gr
import numpy as np
import joblib
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tensorflow as tf

# Load model artifacts
print("Loading model artifacts...")
model_artifacts = joblib.load('svm_vgg16_cats_dogs.pkl')

svm_model = model_artifacts['svm_model']
scaler = model_artifacts['scaler']
pca = model_artifacts['pca']
feature_extractor = model_artifacts['feature_extractor']

print("Model loaded successfully!")

IMG_WIDTH = 224
IMG_HEIGHT = 224

def predict_image(image):
    """
    Predict whether the image contains a cat or dog
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        dict: Prediction results with probabilities
    """
    try:
        # Ensure image is PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize and preprocess
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Extract features using VGG16
        features = feature_extractor.predict(img_array, verbose=0)
        
        # Scale and apply PCA
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Make prediction
        prediction = svm_model.predict(features_pca)[0]
        probabilities = svm_model.predict_proba(features_pca)[0]
        
        # Prepare results
        result = {
            'Cat': float(probabilities[0]),
            'Dog': float(probabilities[1])
        }
        
        predicted_class = 'Cat' if prediction == 0 else 'Dog'
        confidence = probabilities[prediction]
        
        return result, f"**Prediction: {predicted_class}** (Confidence: {confidence:.2%})"
        
    except Exception as e:
        return {}, f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload an image of a cat or dog"),
    outputs=[
        gr.Label(num_top_classes=2, label="Classification Probabilities"),
        gr.Textbox(label="Prediction Result")
    ],
    title="🐱🐶 Dogs vs Cats Classifier",
    description="""
    ### SVM + VGG16 Transfer Learning Model
    
    This model uses **VGG16** for feature extraction combined with **Support Vector Machine (SVM)** 
    for classification. Upload an image of a cat or dog to get predictions!
    
    **Model Details:**
    - Feature Extraction: VGG16
    - Dimensionality Reduction: PCA (512 components)
    - Classifier: SVM with RBF kernel
    - Validation Accuracy: ~98%+
    
    **Try it out with your own images!**
    """,
    examples=[
        # Add example images here if you have them
    ],
    article="""
    ### About This Model
    
    This classifier was trained using transfer learning, leveraging VGG16's powerful feature 
    extraction capabilities combined with the discriminative power of Support Vector Machines.
    
    **Technical Details:**
    - Images are resized to 224x224 pixels
    - Features are extracted using VGG16 (512-dimensional vectors)
    - StandardScaler normalizes features
    - PCA reduces dimensionality while preserving variance
    - SVM with RBF kernel performs final classification
    
    **Performance Metrics:**
    - Accuracy: ~98%+
    - Precision: >0.95
    - Recall: >0.95
    - F1-Score: >0.95
    
    **Repository:** [GitHub](https://github.com/990aa/SCT_ML_3)
    
    ---
    """,
    theme="default",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
