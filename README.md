---
title: Dogs vs Cats Classifier (SVM + VGG16)
emoji: 🐱🐶
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
tags:
  - image-classification
  - computer-vision
  - svm
  - transfer-learning
  - vgg16
  - dogs
  - cats
  - sklearn
datasets:
  - kaggle/dogs-vs-cats
metrics:
  - accuracy
  - precision
  - recall
  - f1
---

# 🐱🐶 Dogs vs Cats Classifier

A binary image classification model that distinguishes between cats and dogs using **Support Vector Machine (SVM)** combined with **VGG16** transfer learning.

## Model Description

This model implements a hybrid approach combining classical machine learning with deep learning:

- **Feature Extraction**: VGG16 (pre-trained on ImageNet) extracts 512-dimensional feature vectors
- **Dimensionality Reduction**: PCA maintains 512 components while decorrelating features
- **Classification**: SVM with RBF kernel performs the final binary classification

### Model Architecture

```
Input Image (224x224x3)
    ↓
VGG16 Feature Extractor (frozen)
    ↓
512-dimensional features
    ↓
StandardScaler (normalization)
    ↓
PCA (512 components)
    ↓
SVM (RBF kernel)
    ↓
Binary Output (Cat=0, Dog=1)
```

## Intended Use

### Primary Use Cases
- Binary classification of cat and dog images
- Educational demonstration of transfer learning
- Baseline model for image classification tasks
- Feature extraction pipeline for similar datasets

### Out-of-Scope Use Cases
- Multi-class classification (only cat/dog supported)
- Real-time video classification (not optimized for speed)
- Classification of other animals or objects
- Medical or safety-critical applications

## Training Data

- **Dataset**: Kaggle Dogs vs Cats Dataset
- **Training Samples**: 2,000 images (1,000 cats, 1,000 dogs)
- **Validation Samples**: 500 images (250 cats, 250 dogs)
- **Image Format**: JPEG, various resolutions (resized to 224x224)
- **Preprocessing**: VGG16-specific normalization (ImageNet mean subtraction)

## Performance Metrics

### Validation Set Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~98%+ |
| **Precision** | >0.95 |
| **Recall** | >0.95 |
| **F1-Score** | >0.95 |

### Confusion Matrix

The model shows balanced performance across both classes with minimal misclassifications.

### Class-Specific Performance

- **Cat Classification**: High accuracy with low false positive rate
- **Dog Classification**: High accuracy with low false negative rate

## Hyperparameters

### SVM Configuration
- **Kernel**: RBF (Radial Basis Function)
- **C**: Optimized via GridSearchCV
- **Gamma**: Optimized via GridSearchCV
- **Probability**: True (for confidence scores)

### Feature Extraction
- **Model**: VGG16
- **Weights**: ImageNet pre-trained
- **Pooling**: Global Average Pooling
- **Output Dimension**: 512

### Dimensionality Reduction
- **Method**: PCA
- **Components**: 512
- **Explained Variance**: >95%

## Limitations

### Known Limitations

1. **Dataset Size**: Trained on a subset (2,000 images) of the full dataset
2. **Binary Classification Only**: Cannot distinguish between breeds or other animals
3. **Image Quality**: Performance may degrade with low-quality or heavily occluded images
4. **Bias**: May inherit biases from the VGG16 pre-training (ImageNet)
5. **Computational Cost**: Requires TensorFlow for feature extraction


## Training Procedure

### Preprocessing

1. **Image Loading**: Resize to 224x224 pixels
2. **Normalization**: VGG16 preprocessing (ImageNet mean subtraction)
3. **Feature Extraction**: VGG16 forward pass (512-d vectors)
4. **Standardization**: Zero mean, unit variance
5. **PCA**: Dimensionality reduction to 512 components

### Training

1. **Hyperparameter Search**: GridSearchCV with 3-fold cross-validation
2. **Parameter Grid**:
   - C: [0.1, 1, 10, 100]
   - Gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
   - Kernel: ['rbf', 'linear']
3. **Optimization**: Sequential Minimal Optimization (SMO)
4. **Validation**: Stratified train-test split (80/20)

## Evaluation

### Methodology

- **Split**: 80% train, 20% validation
- **Stratification**: Balanced class distribution
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Confusion matrix, prediction samples

### Results Analysis

- High accuracy across both classes
- Minimal class imbalance
- Strong generalization to validation set
- Low variance in cross-validation scores


## Additional Information

### Repository
- **GitHub**: [https://github.com/990aa/SCT_ML_3](https://github.com/990aa/SCT_ML_3)
- **Technical Report**: See repository for detailed mathematical concepts

### License
This model is released under the MIT License.
---
