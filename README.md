# 🎙️ Speech Emotion Recognition using Deep Learning

A comprehensive AI-based system for recognizing human emotions from speech audio using state-of-the-art deep learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Web Application](#web-application)
- [Screenshots](#screenshots)

## 🎯 Overview

This project implements a Speech Emotion Recognition (SER) system that can classify emotions from speech into 8 categories:

- 😐 **Neutral**
- 😌 **Calm**
- 😊 **Happy**
- 😢 **Sad**
- 😠 **Angry**
- 😨 **Fear**
- 🤢 **Disgust**
- 😲 **Surprise**

## ✨ Features

### Core Features
- **Multiple Model Architectures**: CNN, LSTM, and hybrid CNN+LSTM models
- **Advanced Audio Processing**: Silence removal, noise reduction, normalization
- **Rich Feature Extraction**: MFCC, Mel Spectrogram, Chroma, ZCR, RMS, Spectral Centroid
- **High Accuracy**: Achieves excellent performance on standard datasets
- **Multiple Dataset Support**: RAVDESS, TESS, and EMO-DB

### Web Application Features
- 🎨 **Modern UI**: Beautiful gradient theme with responsive design
- 📁 **File Upload**: Easy audio file upload (.wav format)
- 🎤 **Voice Recording**: Record audio directly (browser-dependent)
- 📊 **Visualizations**: Waveform, Spectrogram, MFCC heatmaps
- 📈 **Probability Charts**: Interactive emotion probability distributions
- 📜 **Prediction History**: Track all predictions in session
- 💾 **Download Reports**: Export prediction results
- 🌙 **Dark Mode**: Toggle between light and dark themes
- ⚙️ **Model Selection**: Switch between different model types

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | TensorFlow, Keras |
| **Audio Processing** | Librosa, SoundFile |
| **Scientific Computing** | NumPy, Pandas, SciPy |
| **Machine Learning** | Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Web Interface** | Streamlit |
| **Development** | Python 3.11+ |

## 📁 Project Structure

```
SpeechEmotionRecognition/
│
├── dataset/                      # Dataset directory
│   ├── RAVDESS/                 # RAVDESS dataset
│   ├── TESS/                    # TESS dataset
│   └── EMO_DB/                  # EMO-DB dataset
│
├── models/                       # Model architectures
│   ├── __init__.py
│   ├── cnn_model.py             # CNN architecture
│   ├── lstm_model.py            # LSTM architecture
│   └── cnn_lstm_model.py        # Hybrid CNN+LSTM (best)
│
├── saved_models/                 # Trained models & results
│   ├── best_model_*.h5          # Best model checkpoints
│   ├── final_model_*.h5         # Final trained models
│   ├── metrics_*.json           # Performance metrics
│   ├── training_history_*.png   # Training curves
│   ├── confusion_matrix_*.png   # Confusion matrices
│   └── features_cache.pkl       # Cached features
│
├── utils/                        # Utility modules
│   ├── __init__.py
│   ├── audio_processing.py      # Audio preprocessing
│   ├── feature_extraction.py    # Feature extraction
│   ├── dataset_loader.py        # Dataset loading
│   └── visualization.py         # Plotting utilities
│
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── app.py                        # Streamlit web app
├── config.py                     # Configuration settings
├── requirements.txt              # Dependencies
├── labels.json                   # Emotion labels
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Step 1: Clone or Download

```bash
cd SpeechEmotionRecognition
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; import librosa; print('Installation successful!')"
```

## 📊 Dataset Setup

### Supported Datasets

1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - Download from: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)
   - Contains 24 professional actors (12 male, 12 female)
   - 7,356 files with emotional expressions

2. **TESS** (Toronto Emotional Speech Set)
   - Download from: [https://tspace.library.utoronto.ca/handle/1807/24487](https://tspace.library.utoronto.ca/handle/1807/24487)
   - Contains 2 female actors
   - 2,800 files with 7 emotions

3. **EMO-DB** (Berlin Database of Emotional Speech)
   - Download from: [http://emodb.bilderbar.info/download/](http://emodb.bilderbar.info/download/)
   - Contains 10 actors (5 male, 5 female)
   - 535 files with German speech

### Directory Structure

Place downloaded datasets in the following structure:

```
dataset/
├── RAVDESS/
│   ├── Actor_01/
│   │   ├── 03-01-01-01-01-01-01.wav
│   │   └── ...
│   └── Actor_02/
│       └── ...
│
├── TESS/
│   ├── OAF_angry/
│   │   ├── OAF_angry_001.wav
│   │   └── ...
│   └── YAF_happy/
│       └── ...
│
└── EMO_DB/
    ├── 03a01Fa.wav
    ├── 03a01Wa.wav
    └── ...
```

**Note**: The system automatically detects which datasets are available and loads them.

## 🎓 Usage

### 1. Training a Model

Train the default CNN+LSTM model:

```bash
python train.py
```

Train a specific model:

```bash
# CNN model
python train.py --model cnn

# LSTM model
python train.py --model lstm

# CNN+LSTM hybrid (recommended)
python train.py --model cnn_lstm
```

Force re-extraction of features (ignore cache):

```bash
python train.py --no-cache
```

### Training Output

The training script will:
- Load and preprocess all available datasets
- Extract audio features (MFCC, Mel spectrogram, etc.)
- Split data into train/validation/test sets (80%/10%/10%)
- Train the model with early stopping
- Save the best model checkpoint
- Generate performance visualizations
- Display evaluation metrics

### 2. Making Predictions

Predict emotion from a single audio file:

```bash
python predict.py path/to/audio.wav
```

Specify model type:

```bash
python predict.py audio.wav --model-type cnn_lstm
```

Use a specific model file:

```bash
python predict.py audio.wav --model saved_models/best_model_cnn_lstm.h5
```

Save prediction to JSON:

```bash
python predict.py audio.wav --output result.json
```

### Prediction Output

```
============================================================
🎙️  Emotion Prediction Result
============================================================
Predicted Emotion: HAPPY 😊
Confidence: 87.45%

Probability Distribution:
------------------------------------------------------------
happy      😊 │██████████████████████████████████████████████│ 87.45%
neutral    😐 │████████                                      │ 5.23%
calm       😌 │███                                           │ 2.34%
surprise   😲 │██                                            │ 1.89%
sad        😢 │█                                             │ 1.45%
angry      😠 │█                                             │ 0.98%
fear       😨 │                                              │ 0.44%
disgust    🤢 │                                              │ 0.22%
============================================================
```

### 3. Running the Web Application

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Web App Features

1. **Home Page** 🏠
   - Project overview
   - Supported emotions
   - Quick start guide

2. **Predict Emotion** 🎯
   - Upload audio files (.wav)
   - Real-time emotion prediction
   - Interactive visualizations (waveform, spectrogram, MFCC)
   - Probability distribution charts
   - Prediction history
   - Download reports

3. **Model Performance** 📊
   - Training/validation curves
   - Confusion matrices
   - Classification reports
   - Model metrics

4. **About** ℹ️
   - Technical details
   - Architecture overview
   - Dataset information
   - Usage instructions

## 🏗️ Model Architecture

### Hybrid CNN + LSTM Model (Recommended)

```
Input (MFCC Features)
         ↓
   Conv2D (64 filters)
         ↓
  BatchNormalization
         ↓
    MaxPooling2D
         ↓
      Dropout
         ↓
   Conv2D (128 filters)
         ↓
  BatchNormalization
         ↓
    MaxPooling2D
         ↓
      Dropout
         ↓
      Reshape
         ↓
Bidirectional LSTM (128)
         ↓
      Dropout
         ↓
   Dense (128, ReLU)
         ↓
      Dropout
         ↓
Dense (8, Softmax)
         ↓
  Emotion Prediction
```

### Key Features

- **CNN Layers**: Extract spatial patterns from MFCC spectrograms
- **Bidirectional LSTM**: Capture temporal dependencies in both directions
- **Batch Normalization**: Stabilize and accelerate training
- **Dropout**: Prevent overfitting
- **L2 Regularization**: Additional overfitting prevention

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 50 (with early stopping) |
| Loss Function | Sparse Categorical Crossentropy |
| Early Stopping Patience | 10 epochs |
| Validation Split | 10% |
| Test Split | 10% |

## 📈 Results

### Model Performance Comparison

| Model | Test Accuracy | Test Loss | Parameters |
|-------|---------------|-----------|------------|
| CNN | ~75-80% | ~0.65 | ~500K |
| LSTM | ~70-75% | ~0.75 | ~300K |
| **CNN+LSTM** | **~80-85%** | **~0.55** | **~600K** |

*Note: Actual results depend on dataset size and composition*

### Feature Importance

The model primarily relies on:
1. **MFCC coefficients** (40 features)
2. **Mel Spectrogram** (128 mel bands)
3. **Temporal patterns** (captured by LSTM)

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Emotion-specific precision scores
- **Recall**: Emotion-specific recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

## 🖼️ Screenshots

### Web Application

*[Add screenshots of your Streamlit app here]*

**Home Page**
- Clean, modern interface with gradient design
- Quick start guide
- Supported emotions display

**Prediction Page**
- File upload interface
- Real-time emotion analysis
- Interactive visualizations
- Probability charts

**Performance Page**
- Training curves
- Confusion matrix
- Detailed metrics

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Audio preprocessing
SAMPLE_RATE = 22050
DURATION = 3.0
N_MFCC = 40

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Emotions
EMOTIONS = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    # ...
}
```

## 🐛 Troubleshooting

### Issue: Model not found

**Solution**: Train a model first using `python train.py`

### Issue: Audio file format error

**Solution**: Ensure audio is in `.wav` format. Convert using:

```bash
ffmpeg -i input.mp3 output.wav
```

### Issue: Low accuracy

**Solutions**:
- Ensure sufficient dataset size
- Check audio quality
- Try data augmentation
- Increase training epochs
- Tune hyperparameters

### Issue: Out of memory

**Solutions**:
- Reduce batch size in `config.py`
- Use feature caching
- Close other applications

## 📝 Future Enhancements

- [ ] Real-time audio recording in web app
- [ ] Support for more audio formats (mp3, m4a, etc.)
- [ ] Multi-language support
- [ ] Transfer learning from pre-trained models
- [ ] Ensemble model predictions
- [ ] Mobile app deployment
- [ ] REST API for predictions
- [ ] Docker containerization

## 📚 References

1. Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
2. Pichora-Fuller MK, Dupuis K (2020) Toronto emotional speech set (TESS)
3. Burkhardt F, et al. (2005) A database of German emotional speech
4. Librosa Documentation: [https://librosa.org](https://librosa.org)
5. TensorFlow Documentation: [https://tensorflow.org](https://tensorflow.org)

## 📄 License

This project is developed for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 👨‍💻 Author

Created for Final Year Engineering Project

## 🙏 Acknowledgments

- Dataset creators: RAVDESS, TESS, EMO-DB teams
- TensorFlow and Keras development teams
- Librosa audio processing library
- Streamlit web framework
- Open-source community

---

**Made with ❤️ using Python, TensorFlow, and Streamlit**

For questions or issues, please open an issue on the project repository.
