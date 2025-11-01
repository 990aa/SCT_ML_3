# Dogs vs Cats Classification using SVM and VGG16

## Project Overview
This project implements a binary image classification system to distinguish between cats and dogs using Support Vector Machines (SVM) combined with deep learning feature extraction via VGG16. The approach leverages transfer learning to extract meaningful features from images, which are then classified using a traditional machine learning algorithm.

## Dataset
- **Source**: Kaggle Dogs vs Cats Dataset
- **Training Images**: 25,000 labeled images (12,500 cats, 12,500 dogs)
- **Test Images**: 12,500 unlabeled images
- **Image Format**: JPEG files with varying dimensions

## Methodology

### 1. Feature Extraction
- **Input Size**: 224x224 pixels (RGB)
- **Feature Vector**: 512-dimensional features extracted from the last pooling layer
- **Preprocessing**: Images are preprocessed using VGG16's preprocessing function

### 2. Dimensionality Reduction
- **Technique**: Principal Component Analysis (PCA)
- **Target Dimensions**: 512 components
- **Purpose**: Reduce computational complexity while preserving variance

### 3. Feature Standardization
- **Method**: StandardScaler (zero mean, unit variance)
- **Applied**: Before PCA transformation

### 4. Classification
- **Algorithm**: Support Vector Machine (SVM)
- **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation
- **Parameters Tuned**:
  - C (regularization): [0.1, 1, 10, 100]
  - Gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
  - Kernel: ['rbf', 'linear']

## Results

### Model Performance
- **Validation Accuracy**: ~98%
- **Precision**: High (~99%)
- **Recall**: High (~97%)
- **F1-Score**: High (~98%)

### Best Hyperparameters
The optimal SVM configuration was determined through cross-validation.

## Project Structure
```
cats_dogs_svm/
│
├── dogs-vs-cats.ipynb      # Main implementation notebook
├── README.md               # Project documentation
├── TECHNICAL_REPORT.md     # Detailed mathematical concepts
├── pyproject.toml          # Project dependencies
└── svm_vgg16_cats_dogs.pkl # Saved model artifacts
```

## Dependencies
- **Python**: 3.10+
- **Key Libraries**:
  - scikit-learn: Machine learning algorithms
  - TensorFlow/Keras: Deep learning framework
  - NumPy: Numerical computations
  - Pandas: Data manipulation
  - Matplotlib & Seaborn: Visualization
  - OpenCV & scikit-image: Image processing
  - tqdm: Progress bars

## Key Features
- ✅ Transfer learning with VGG16
- ✅ Efficient feature extraction
- ✅ Dimensionality reduction with PCA
- ✅ Hyperparameter optimization
- ✅ Comprehensive evaluation metrics
- ✅ Visualization of results and misclassifications
- ✅ Model persistence for deployment

## Performance Considerations
- **Training Subset**: Uses 2,000 training images for faster execution
- **Validation Subset**: Uses 500 validation images
- **Batch Processing**: Images processed in batches for efficiency
- **Memory Management**: Features extracted and processed in manageable chunks

## Visualization
The notebook includes:
- Confusion matrix heatmap
- Sample predictions with confidence scores
- Misclassified examples analysis
- Error rate breakdown by class