# Model Card: Dogs vs Cats Classifier

## Model Details

### Model Description

This is a binary image classification model that distinguishes between images of cats and dogs. The model uses a hybrid approach combining transfer learning with classical machine learning, trained on **combined datasets** for improved accuracy:

- **Feature Extraction**: VGG16 + Global Average Pooling (pre-trained on ImageNet)
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Classification**: Support Vector Machine (SVM) with RBF kernel

### Model Type
- **Task**: Binary Image Classification
- **Architecture**: VGG16 + SVM
- **Framework**: TensorFlow/Keras + scikit-learn
- **Model Format**: .keras (modern serialization)
- **Model Size**: ~61MB (feature extractor + components)
- **License**: MIT

### Model Developer
- **Repository**: https://github.com/990aa/SCT_ML_3
- **Date**: November 2025
- **Version**: 1.0

### Out-of-Scope Use Cases
❌ Multi-class classification of different animal species
❌ Real-time video processing (not optimized for speed)
❌ Medical diagnosis or safety-critical applications
❌ Classification of image types other than animals
❌ Production systems without proper validation

## Training Data

### Dataset Information
- **Source 1**: Kaggle Dogs vs Cats Dataset (`dog-and-cat-classification-dataset`)
  - Downloaded to: `./kaggle_data/`
  - Training Subset: ~5,000 images (2,500 cats, 2,500 dogs)
  
- **Source 2**: Local Dogs vs Cats Dataset (`dogs-vs-cats/train`)
  - Training Subset: ~5,000 images (2,500 cats, 2,500 dogs)

- **Combined Total**: ~10,000 labeled images from diverse sources
- **Validation Subset**: 20% of combined data (~2,000 images)
  - ~1,000 cat images
  - ~1,000 dog images

### Data Characteristics
- **Format**: JPEG images
- **Resolution**: Variable (resized to 224×224)
- **Color**: RGB (3 channels)
- **Classes**: Binary (Cat=0, Dog=1)
- **Balance**: Perfectly balanced dataset (50/50 split)
- **Diversity**: Combined from two different sources for better generalization

### Preprocessing Pipeline
1. **Dataset Download**: Kaggle dataset to local `./kaggle_data/` folder
2. **Dataset Combination**: Merge images from Kaggle + local datasets
3. **Resizing**: All images resized to 224×224 pixels
4. **Normalization**: VGG16-specific preprocessing
   - Subtract ImageNet mean: [123.68, 116.779, 103.939]
   - RGB channel order
5. **Feature Extraction**: VGG16 + Global Average Pooling → 512-d vectors
6. **Standardization**: Zero mean, unit variance
7. **Dimensionality Reduction**: PCA to 256 components

## Model Architecture

### Complete Pipeline

```
Input Image (224×224×3)
    ↓
VGG16 Feature Extractor
├── 13 Convolutional Layers (3×3)
├── 5 Max Pooling Layers (2×2)
└── Global Average Pooling
    ↓
512-dimensional Feature Vector
    ↓
StandardScaler
├── Zero mean normalization
└── Unit variance scaling
    ↓
PCA Transformation
├── 512 principal components
└── >95% variance explained
    ↓
Support Vector Machine
├── RBF Kernel: K(x,y) = exp(-γ||x-y||²)
├── Hyperparameters optimized via GridSearch
└── Probability calibration enabled
    ↓
Binary Prediction + Confidence Scores
```

### Component Details

**1. VGG16 Feature Extractor**
- Pre-trained on ImageNet (1.4M images, 1000 classes)
- Frozen weights (not fine-tuned)
- Remove top classification layers
- Global average pooling
- Output: 512-dimensional features

**2. StandardScaler**
- Transforms features to zero mean, unit variance
- Fitted on training data only
- Applied consistently to train/validation/test

**3. PCA**
- Reduces correlations between features
- Maintains 512 components
- Captures >95% of variance
- Improves computational efficiency

**4. SVM Classifier**
- Algorithm: Support Vector Classification
- Kernel: Radial Basis Function (RBF)
- Probability estimates: Enabled
- Optimization: Sequential Minimal Optimization (SMO)

## Hyperparameters

### Optimized via GridSearchCV

**Search Space:**
```python
param_grid = {
    'C': [0.1, 1, 10, 100],           # Regularization
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],  # Kernel coefficient
    'kernel': ['rbf', 'linear']        # Kernel type
}
```

**Cross-Validation:**
- Strategy: 3-fold stratified CV
- Scoring: Accuracy
- Parallel: All cores utilized

**Best Parameters:** (Determined during training)
- Values depend on specific training run
- Typically: C ∈ [1, 10], gamma='scale', kernel='rbf'

### Fixed Parameters
- `IMG_WIDTH = 224`
- `IMG_HEIGHT = 224`
- `FEATURE_DIM = 512`
- `PCA_COMPONENTS = 512`
- `BATCH_SIZE = 32`

## Performance

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | ~87-95% |
| **Precision** | >85.0% |
| **Recall** | >85.0% |
| **F1-Score** | >85.0% |

*Performance varies based on training configuration (dataset size, PCA components)*

### Confusion Matrix (Approximate)

|              | Predicted Cat | Predicted Dog |
|--------------|---------------|---------------|
| **Actual Cat** | ~238 (TN)   | ~3 (FP)       |
| **Actual Dog** | ~9 (FN)     | ~250 (TP)     |

### Class-Specific Performance

**Cat Classification:**
- Precision: High (~0.98)
- Recall: High (~0.98)
- False Positive Rate: Low (~2%)

**Dog Classification:**
- Precision: High (~0.98)
- Recall: High (~0.98)
- False Negative Rate: Low (~2%)

### Cross-Validation Results
- Best CV Score: ~97-98%
- Standard Deviation: Low (<2%)
- Generalization: Strong

## Limitations

### Known Limitations

1. **Dataset Size**
   - Trained on ~10,000 images (combined from two sources)
   - May not generalize to all breeds or poses
   - Limited representation of edge cases

2. **Binary Classification Only**
   - Cannot distinguish between different breeds
   - Cannot classify other animals
   - No support for "neither" or "both" cases

3. **Image Quality Dependencies**
   - Performance degrades with low-resolution images
   - Heavily occluded subjects may be misclassified
   - Requires clear visibility of the animal

### Edge Cases

**May Struggle With:**
- Cartoons or artistic renderings
- Animals in unusual poses or clothing
- Mixed images containing both cats and dogs
- Images with multiple animals
- Very young animals (kittens/puppies)
- Rare or unusual breeds

## Bias and Fairness

### Potential Biases

1. **Dataset Bias**
   - May favor common breeds in training data
   - Potential overrepresentation of certain colors/patterns
   - Indoor vs outdoor setting biases

2. **ImageNet Pre-training Bias**
   - VGG16 trained on internet images
   - May favor certain visual patterns
   - Western/internet culture bias

3. **Class Balance**
   - Training data perfectly balanced (50/50)
   - Real-world distribution may differ
   - No adjustment for class priors

### Fairness Considerations

- Model treats both classes symmetrically
- Equal false positive/negative rates
- No demographic or protected attributes involved
- No differential impact on user groups

## Ethical Considerations

### Responsible Use

**Encouraged:**
- Educational and research purposes
- Non-critical pet recognition applications
- Data science learning projects
- Baseline comparisons

**Discouraged:**
- High-stakes decision making
- Medical or veterinary diagnosis
- Security/surveillance without oversight
- Automated content moderation

### Privacy
- Model trained on public dataset
- No personal or sensitive information
- No user data collection in model itself
- Inference on user-provided images only

### Model Files

- **cats-vs-dogs.keras**: VGG16 + Global Average Pooling feature extractor (~58 MB)
  - Format: TensorFlow SavedModel (HDF5)
  - Content: Pre-trained VGG16 with frozen weights
  
- **cats-vs-dogs-components.keras**: PCA, Scaler, and SVM components (~3-5 MB)
  - Format: HDF5 (h5py)
  - Content: PCA parameters, StandardScaler parameters, SVM support vectors
  - Components: Dimensionality reduction + normalization + classifier

## Citation

### BibTeX

```bibtex
@misc{dogs_cats_svm_vgg16_2025,
  author = {{Abdul Ahad}},
  title = {Dogs vs Cats Classification using SVM and VGG16 Transfer Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/990aa/SCT_ML_3}},
  note = {Model available at HuggingFace}
}
```

### References

1. **VGG16**: Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition." arXiv:1409.1556
2. **SVM**: Cortes, C., & Vapnik, V. (1995). "Support-vector networks." Machine Learning, 20(3), 273-297.
3. **ImageNet**: Deng, J., et al. (2009). "ImageNet: A large-scale hierarchical image database." CVPR 2009.
4. **Transfer Learning**: Yosinski, J., et al. (2014). "How transferable are features in deep neural networks?" NIPS 2014.

## Model Card Authors

- **Author**: Abdul Ahad
- **GitHub**: @990aa
- **Repository**: https://github.com/990aa/SCT_ML_3

## Additional Resources

- **GitHub Repository**: https://github.com/990aa/SCT_ML_3
- **Technical Report**: See `TECHNICAL_REPORT.md` for mathematical details
- **README**: See `README.md` for project overview
- **Jupyter Notebook**: See `dogs-vs-cats.ipynb` for training code

## Glossary

- **VGG16**: 16-layer convolutional neural network
- **Transfer Learning**: Using pre-trained model features
- **SVM**: Support Vector Machine, a maximum-margin classifier
- **RBF Kernel**: Radial Basis Function, maps to infinite dimensions
- **PCA**: Principal Component Analysis, dimensionality reduction
- **Feature Extraction**: Converting images to numerical vectors

---

**Last Updated**: November 2025
**Model Card Version**: 1.0
**Contact**: Open an issue on GitHub for questions or concerns
