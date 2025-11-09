---
license: mit
tags:
  - image-classification
  - computer-vision
  - cats-vs-dogs
  - svm
  - vgg16
  - transfer-learning
  - keras
  - tensorflow
  - scikit-learn
language:
  - en
metrics:
  - accuracy
  - precision
  - recall
library_name: keras
datasets:
  - dogs-vs-cats
---

# 🐱🐶 Dogs vs Cats Classifier

A binary image classification model that distinguishes between cats and dogs using **Support Vector Machine (SVM)** combined with **VGG16** transfer learning. The model is trained on **two combined datasets** and saved in modern **.keras format**.

## Model Description

This model implements a hybrid approach combining classical machine learning with deep learning:

- **Feature Extraction**: VGG16 (pre-trained on ImageNet) with Global Average Pooling → 512 features
- **Dimensionality Reduction**: PCA reduces to 256 components (~95% variance)
- **Classification**: SVM with RBF kernel performs binary classification

### Model Architecture

```
Input Image (224×224×3)
    ↓
VGG16 Feature Extractor (frozen)
    ↓
Global Average Pooling → 512 features
    ↓
StandardScaler (normalization)
    ↓
PCA (512 → 256 components)
    ↓
SVM (RBF kernel)
    ↓
Binary Output (Cat=0, Dog=1)
```

### Model Files

- **cats-vs-dogs-components.keras** (~58 MB): VGG16 + Global Average Pooling feature extractor
- **cats-vs-dogs-components.keras** (~3-5 MB): PCA, StandardScaler, and SVM components stored in HDF5 format

## Quick Start

### Option 1: Use Pre-trained Model (2 minutes)

**Install dependencies:**
```bash
uv pip install tensorflow scikit-learn h5py huggingface-hub pillow
```

**Run inference:**
```bash
uv run python inference.py test/cat1.jpeg
```

**Expected output:**
```
🐱🐶 Dogs vs Cats Classifier - Inference (.keras format)
Image: test/cat1.jpeg

✓ Using local model files

Loading model components...
✓ Feature extractor loaded
✓ PCA, Scaler, and SVM loaded

Making prediction...

Prediction: Cat 🐱
Confidence: 92.34%
```

---

### Option 2: Train Your Own Model (30-60 minutes)

**Install dependencies:**
```bash
uv pip install tensorflow scikit-learn h5py kagglehub pillow jupyter matplotlib seaborn tqdm
```

**Start Jupyter Notebook:**
```bash
jupyter notebook train_model.ipynb
```

**Training process:**
1. Cell 1: Install packages
2. Cell 2: Import libraries
3. Cell 3: Download Kaggle dataset to `./kaggle_data/`
4. Cell 4: Configure training (combines Kaggle + local datasets)
5. Cell 5: Load images from **both datasets**
6. Cells 6-12: Train VGG16 → PCA → SVM pipeline
7. Cell 13: Save as `.keras` format
8. Cell 14: Display summary

**Files generated:**
- `cats-vs-dogs-components.keras` - Feature extractor
- `cats-vs-dogs-components.keras` - PCA/Scaler/SVM components

**Test the model:**
```bash
uv run python inference.py test/dog1.jpeg
```

---

### Option 3: Upload to HuggingFace (5 minutes)

**Create `.env` file:**
```env
HF_TOKEN_CD=hf_your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

**Run upload script:**
```bash
uv run python upload_to_huggingface.py
```

**View your model:**
Visit `https://huggingface.co/YOUR_USERNAME/dogs-vs-cats-svm`

---

## Training Data

### Datasets Used

**1. Kaggle Dataset**: `dog-and-cat-classification-dataset`
- Auto-downloaded to `./kaggle_data/`
- 5,000 samples (2,500 cats, 2,500 dogs)

**2. Local Dataset**: `dogs-vs-cats/train/`
- 5,000 samples (2,500 cats, 2,500 dogs)

**Total Combined**: 10,000 images from diverse sources

### Configuration

Edit `train_model.ipynb` Cell 4 for different configurations:

| Config | Samples | Time | Expected Accuracy |
|--------|---------|------|-------------------|
| **Fast** | 2K (1K each) | 5-10 min | ~75% |
| **Balanced** | 10K (5K each) | 30-45 min | ~87% |
| **Maximum** | 25K (12.5K each) | 60-90 min | ~95% |

```python
N_SAMPLES_PER_DATASET = 5000  # Adjust
PCA_COMPONENTS = 256           # 128 (fast), 256 (balanced), 512 (max)
```

---

## Performance Metrics

### Validation Set Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | ~87-95% |
| **Precision** | >0.85 |
| **Recall** | >0.85 |
| **F1-Score** | >0.85 |


### Model Characteristics

- **Balanced Performance**: Equal accuracy for cats and dogs
- **High Confidence**: Most predictions >85% confidence
- **Robust**: Trained on diverse datasets for better generalization
- **Fast Inference**: <1 second per image on CPU

---

## Project Structure

```
cats_dogs_svm/
├── train_model.ipynb             # Training notebook (multi-dataset)
├── inference.py                  # Inference script (.keras format)
├── upload_to_huggingface.py      # HuggingFace upload script
│
├── cats-vs-dogs-components.keras            # VGG16 feature extractor (~58 MB)
├── cats-vs-dogs-components.keras # PCA/Scaler/SVM (~3-5 MB)
│
├── kaggle_data/                  # Auto-downloaded Kaggle dataset
│   └── dog-and-cat-classification-dataset/
│       └── train/
│
├── dogs-vs-cats/                 # Local dataset
│   └── train/
│
├── test/                         # Test images
│   ├── cat1.jpeg
│   ├── dog1.jpeg
│   └── ...
│
└── model_cache/                  # HuggingFace model cache
```

---

## Training Procedure

### Preprocessing Pipeline

1. **Download**: Kaggle dataset to `./kaggle_data/`
2. **Load**: Combine images from both datasets
3. **Resize**: All images to 224×224 pixels
4. **Normalize**: VGG16 preprocessing (ImageNet mean subtraction)
5. **Extract**: VGG16 forward pass → 512-d feature vectors
6. **Transform**: PCA dimensionality reduction to 256 components
7. **Scale**: StandardScaler for zero mean, unit variance

### Model Training

1. **Train/Validation Split**: 80/20 stratified split
2. **Hyperparameter Tuning**: GridSearchCV with 3-fold CV
3. **Parameter Grid**:
   - C: [1, 10, 100]
   - Gamma: ['scale', 0.001, 0.01]
   - Kernel: ['rbf']
4. **Optimization**: Best parameters selected automatically
5. **Save**: Models in .keras format (VGG16 + components)

---

## Primary Use Cases
✅ Binary classification of cat and dog images  
✅ Educational demonstration of transfer learning  
✅ Baseline model for image classification tasks  
✅ Feature extraction pipeline for similar datasets   

---

## Limitations

### Known Limitations

1. **Binary Classification Only**: Cannot distinguish breeds or other animals
2. **Image Quality Dependent**: Performance degrades with low-quality/occluded images
3. **Dataset Biases**: May inherit biases from ImageNet pre-training
4. **Computational Requirements**: Requires TensorFlow for feature extraction

### May Struggle With

- Cartoons or artistic renderings
- Animals in unusual poses or clothing
- Mixed images containing both cats and dogs
- Very young animals (kittens/puppies)
- Rare or unusual breeds

---

## Troubleshooting

### Import Errors
```bash
uv pip install --upgrade tensorflow scikit-learn h5py
```

### Out of Memory
Reduce samples in `train_model.ipynb` Cell 4:
```python
N_SAMPLES_PER_DATASET = 2000  # Smaller dataset
BATCH_SIZE = 8                 # Smaller batches
```

### Low Confidence
Train with more samples:
```python
N_SAMPLES_PER_DATASET = 10000  # More data
PCA_COMPONENTS = 512           # Keep more features
```

---

## Additional Information

### Repository
- **GitHub**: [https://github.com/990aa/SCT_ML_3](https://github.com/990aa/SCT_ML_3)
- **HuggingFace**: [https://huggingface.co/a-01a/dogs-vs-cats-svm](https://huggingface.co/a-01a/dogs-vs-cats-svm)

### License
This model is released under the MIT License.

### Citation

```bibtex
@misc{dogs_cats_svm_vgg16_2025,
  author = {{Abdul Ahad}},
  title = {Dogs vs Cats Classification using SVM and VGG16 Transfer Learning},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/990aa/SCT_ML_3}}
}
```

---

**Author**: Abdul Ahad (@990aa)  
**Last Updated**: November 2025  
**Model Format**: Keras (.keras) 