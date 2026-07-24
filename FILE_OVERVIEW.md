# 📁 Complete File Overview

This document provides a detailed description of every file in the project.

## 📂 Project Structure

```
SpeechEmotionRecognition/
├── 📁 Core Application Files
├── 📁 Model Definitions
├── 📁 Utility Modules
├── 📁 Data Directories
└── 📁 Documentation
```

---

## 🎯 Core Application Files

### `config.py` (Configuration)
**Purpose**: Central configuration file for all project settings

**Contains**:
- Audio processing parameters (sample rate, duration, MFCC count)
- Training hyperparameters (batch size, epochs, learning rate)
- Emotion labels and mappings
- Directory paths
- Model types
- UI settings

**Key Settings**:
```python
SAMPLE_RATE = 22050
N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 50
EMOTIONS = {"neutral": 0, "calm": 1, ...}
```

**When to Edit**: Modify hyperparameters, change paths, adjust training settings

---

### `train.py` (Training Script)
**Purpose**: Complete training pipeline for emotion recognition models

**Features**:
- Loads multiple datasets (RAVDESS, TESS, EMO-DB)
- Extracts audio features
- Splits data (train/val/test)
- Trains model with callbacks
- Generates evaluation metrics
- Creates visualizations
- Saves model checkpoints

**Usage**:
```bash
python train.py --model cnn_lstm
python train.py --model cnn --no-cache
```

**Output Files**:
- `saved_models/best_model_*.h5` - Best model checkpoint
- `saved_models/metrics_*.json` - Performance metrics
- `saved_models/training_history_*.png` - Training curves
- `saved_models/confusion_matrix_*.png` - Confusion matrix

**Lines of Code**: ~300

---

### `predict.py` (Prediction Script)
**Purpose**: Command-line interface for emotion prediction

**Features**:
- Loads trained model
- Processes audio file
- Extracts features
- Makes prediction
- Displays results with probabilities
- Exports to JSON

**Usage**:
```bash
python predict.py audio.wav
python predict.py audio.wav --model-type cnn_lstm --output result.json
```

**Classes**:
- `EmotionPredictor`: Main prediction class with model loading and inference

**Lines of Code**: ~250

---

### `app.py` (Streamlit Web Application)
**Purpose**: Interactive web interface for emotion recognition

**Pages**:
1. **Home** - Project overview and quick start
2. **Predict Emotion** - Upload audio, view predictions, visualizations
3. **Model Performance** - Training metrics and evaluation
4. **About** - Technical details and documentation

**Features**:
- File upload interface
- Audio playback
- Real-time prediction
- Interactive visualizations (waveform, spectrogram, MFCC)
- Probability charts
- Prediction history
- Downloadable reports
- Dark mode toggle
- Model selection

**UI Components**:
- Custom CSS styling (gradient theme)
- Sidebar navigation
- Cards and metrics
- Progress indicators
- Session state management

**Usage**:
```bash
streamlit run app.py
# Access at http://localhost:8501
```

**Lines of Code**: ~800+

---

### `quick_start.py` (Installation Verification)
**Purpose**: Verify installation and test core functionality

**Tests**:
- Package imports (TensorFlow, Librosa, etc.)
- Directory structure
- Dataset availability
- Audio processing
- Feature extraction
- Model building

**Usage**:
```bash
python quick_start.py
```

**Output**: Comprehensive status report with next steps

**Lines of Code**: ~300

---

## 🧠 Model Definitions (`models/`)

### `models/__init__.py`
**Purpose**: Package initialization and exports

**Exports**:
- `build_cnn_model`
- `build_lstm_model`
- `build_cnn_lstm_model`

---

### `models/cnn_model.py`
**Purpose**: Convolutional Neural Network architecture

**Architecture**:
```
Conv2D(64) → BN → ReLU → MaxPool → Dropout
Conv2D(128) → BN → ReLU → MaxPool → Dropout
Conv2D(256) → BN → ReLU → MaxPool → Dropout
Flatten → Dense(256) → Dense(128) → Dense(8)
```

**Input**: (n_mfcc, time, 1)
**Output**: 8 emotion probabilities
**Parameters**: ~500K

**Best For**: Spatial pattern recognition in spectrograms

---

### `models/lstm_model.py`
**Purpose**: Long Short-Term Memory network for temporal patterns

**Architecture**:
```
Bidirectional LSTM(128) → Dropout
Bidirectional LSTM(64) → Dropout
Dense(128) → BN → Dense(8)
```

**Input**: (time, n_mfcc)
**Output**: 8 emotion probabilities
**Parameters**: ~300K

**Best For**: Temporal sequence modeling in speech

---

### `models/cnn_lstm_model.py` ⭐
**Purpose**: Hybrid CNN+LSTM architecture (best performance)

**Architecture**:
```
Conv2D(64) → BN → MaxPool → Dropout
Conv2D(128) → BN → MaxPool → Dropout
Reshape to (time, features)
Bidirectional LSTM(128) → Dropout
Dense(128) → Dense(8)
```

**Input**: (n_mfcc, time, 1)
**Output**: 8 emotion probabilities
**Parameters**: ~600K

**Best For**: Combining spatial and temporal features

**Performance**: 80-85% accuracy (best of the three)

---

## 🛠️ Utility Modules (`utils/`)

### `utils/__init__.py`
**Purpose**: Package initialization with all utility exports

---

### `utils/audio_processing.py`
**Purpose**: Audio preprocessing functions

**Functions**:
- `preprocess_audio()` - Complete preprocessing pipeline
- `remove_silence()` - Trim silence from audio
- `normalize_audio()` - Normalize amplitude to [-1, 1]
- `reduce_noise()` - Spectral gating noise reduction
- `resample_audio()` - Resample to target sample rate
- `pad_or_truncate()` - Fixed-length audio
- `augment_audio()` - Data augmentation (time stretch, pitch shift, noise)

**Preprocessing Pipeline**:
```
Load → Resample → Remove Silence → Reduce Noise → Normalize → Pad/Truncate
```

**Lines of Code**: ~200

---

### `utils/feature_extraction.py`
**Purpose**: Extract audio features for model input

**Functions**:
- `extract_mfcc()` - MFCC features (40 coefficients)
- `extract_mel_spectrogram()` - Mel spectrogram (128 bands)
- `extract_chroma()` - Chroma features (12 bins)
- `extract_zcr()` - Zero crossing rate
- `extract_rms()` - RMS energy
- `extract_spectral_centroid()` - Spectral centroid
- `extract_all_features()` - Combined feature extraction
- `extract_features_for_cnn()` - CNN-specific format
- `extract_features_for_lstm()` - LSTM-specific format
- `compute_statistics()` - Statistical aggregations

**Features Extracted**: 183 features per time frame

**Lines of Code**: ~250

---

### `utils/dataset_loader.py`
**Purpose**: Load and parse audio datasets

**Functions**:
- `load_all_datasets()` - Detect and load all available datasets
- `load_ravdess()` - Parse RAVDESS dataset
- `load_tess()` - Parse TESS dataset
- `load_emodb()` - Parse EMO-DB dataset
- `prepare_dataset()` - Extract features from all audio files

**Features**:
- Automatic dataset detection
- Emotion label mapping
- Feature caching for speed
- Progress tracking with tqdm

**Supported Datasets**:
- RAVDESS: 7,356 files, 24 actors
- TESS: 2,800 files, 2 actors
- EMO-DB: 535 files, 10 actors

**Lines of Code**: ~350

---

### `utils/visualization.py`
**Purpose**: Generate plots and visualizations

**Functions**:
- `plot_waveform()` - Audio waveform plot
- `plot_spectrogram()` - Frequency spectrogram
- `plot_mfcc()` - MFCC heatmap
- `plot_training_history()` - Training/validation curves
- `plot_confusion_matrix()` - Confusion matrix heatmap
- `plot_emotion_distribution()` - Dataset distribution
- `plot_prediction_probabilities()` - Probability bar chart
- `generate_classification_report_text()` - Text report
- `save_figure()` - Save matplotlib figure

**Libraries Used**: Matplotlib, Seaborn, Librosa.display

**Lines of Code**: ~250

---

## 📁 Data Directories

### `dataset/`
**Purpose**: Store audio datasets

**Subdirectories**:
- `RAVDESS/` - RAVDESS dataset files
- `TESS/` - TESS dataset files
- `EMO_DB/` - EMO-DB dataset files

**Note**: Datasets must be downloaded separately (see README.md)

---

### `saved_models/`
**Purpose**: Store trained models and outputs

**Contents** (after training):
- `best_model_*.h5` - Best model checkpoints
- `final_model_*.h5` - Final models
- `metrics_*.json` - Performance metrics
- `training_history_*.png` - Training curves
- `confusion_matrix_*.png` - Confusion matrices
- `classification_report_*.txt` - Detailed reports
- `features_cache.pkl` - Cached features (speeds up training)

---

## 📚 Documentation Files

### `README.md` (Main Documentation)
**Purpose**: Complete project documentation

**Sections**:
- Project overview
- Features
- Technology stack
- Installation instructions
- Dataset setup
- Usage guide
- Model architectures
- Results
- Troubleshooting

**Length**: ~500 lines

---

### `INSTALL.md` (Installation Guide)
**Purpose**: Detailed installation instructions

**Sections**:
- Prerequisites
- Step-by-step installation
- System requirements
- Common issues
- GPU setup
- Troubleshooting

**Length**: ~400 lines

---

### `GETTING_STARTED.md` (Quick Start Guide)
**Purpose**: Get up and running quickly

**Sections**:
- 5-minute quick start
- Detailed walkthrough
- First training run
- First prediction
- Using the web app
- Common tasks
- Next steps

**Length**: ~350 lines

---

### `PROJECT_SUMMARY.md` (Technical Overview)
**Purpose**: Comprehensive technical summary

**Sections**:
- Architecture overview
- File structure
- Supported emotions
- Technical implementation
- Model architectures
- Training configuration
- Expected performance
- Key features

**Length**: ~700 lines

---

### `FILE_OVERVIEW.md` (This File)
**Purpose**: Detailed description of every project file

---

## 🔧 Configuration Files

### `requirements.txt`
**Purpose**: Python package dependencies

**Key Packages**:
- tensorflow >= 2.13.0
- librosa >= 0.10.0
- streamlit >= 1.28.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

**Total Packages**: 15+

---

### `labels.json`
**Purpose**: Emotion label definitions

**Content**:
```json
[
  "neutral", "calm", "happy", "sad",
  "angry", "fear", "disgust", "surprise"
]
```

---

### `.gitignore`
**Purpose**: Git version control exclusions

**Excludes**:
- Python cache files (`__pycache__`)
- Virtual environments (`venv/`)
- Large dataset files
- Trained models (`.h5` files)
- Log files

---

## 🚀 Launcher Scripts

### `run_app.bat` (Windows)
**Purpose**: Launch Streamlit app on Windows

**Features**:
- Activates virtual environment
- Checks Streamlit installation
- Launches app with proper settings

**Usage**:
```bash
run_app.bat
```

---

### `run_app.sh` (Linux/Mac)
**Purpose**: Launch Streamlit app on Unix systems

**Features**:
- Same as Windows version
- Unix shell syntax

**Usage**:
```bash
chmod +x run_app.sh
./run_app.sh
```

---

## 📊 File Statistics

| Category | Files | Lines of Code |
|----------|-------|---------------|
| **Core Scripts** | 5 | ~1,650 |
| **Models** | 4 | ~350 |
| **Utils** | 5 | ~1,050 |
| **Documentation** | 6 | ~2,500 |
| **Config** | 3 | ~200 |
| **Total** | **23** | **~5,750** |

---

## 🎯 File Dependencies

### Import Graph

```
config.py
    ↑
    ├── train.py
    ├── predict.py
    ├── app.py
    └── utils/
        ├── audio_processing.py
        ├── feature_extraction.py
        ├── dataset_loader.py → uses audio_processing, feature_extraction
        └── visualization.py

models/
    ├── cnn_model.py
    ├── lstm_model.py
    └── cnn_lstm_model.py
    ↑
    └── train.py (imports all models)

predict.py
    ↑
    └── app.py (uses EmotionPredictor class)
```

---

## 💡 File Modification Guide

### To Change Audio Processing:
**Edit**: `utils/audio_processing.py`
**Also Check**: `config.py` for parameters

### To Modify Models:
**Edit**: `models/cnn_model.py`, `lstm_model.py`, or `cnn_lstm_model.py`
**Also Check**: Input shape requirements

### To Adjust Training:
**Edit**: `train.py` and `config.py`
**Parameters**: Batch size, epochs, learning rate

### To Customize UI:
**Edit**: `app.py`
**Sections**: CSS styling, page layouts, features

### To Add New Dataset:
**Edit**: `utils/dataset_loader.py`
**Add**: New loader function + emotion mapping

### To Change Visualizations:
**Edit**: `utils/visualization.py`
**Libraries**: Matplotlib, Seaborn

---

## 📋 File Checklist

Before running the project, ensure these files exist:

**Core** (Required):
- [x] config.py
- [x] train.py
- [x] predict.py
- [x] app.py

**Models** (Required):
- [x] models/__init__.py
- [x] models/cnn_model.py
- [x] models/lstm_model.py
- [x] models/cnn_lstm_model.py

**Utils** (Required):
- [x] utils/__init__.py
- [x] utils/audio_processing.py
- [x] utils/feature_extraction.py
- [x] utils/dataset_loader.py
- [x] utils/visualization.py

**Data** (User must provide):
- [ ] dataset/RAVDESS/ (download separately)
- [ ] dataset/TESS/ (download separately)
- [ ] dataset/EMO_DB/ (download separately)

**Documentation**:
- [x] README.md
- [x] INSTALL.md
- [x] GETTING_STARTED.md
- [x] PROJECT_SUMMARY.md
- [x] requirements.txt
- [x] labels.json

---

## 🔍 Finding Specific Code

### "Where is the MFCC extraction?"
→ `utils/feature_extraction.py` → `extract_mfcc()`

### "Where is the CNN model defined?"
→ `models/cnn_model.py` → `build_cnn_model()`

### "Where are emotions mapped?"
→ `config.py` → `EMOTIONS` dictionary

### "Where is the training loop?"
→ `train.py` → `train_model()` function

### "Where is the web UI code?"
→ `app.py` → Page sections

### "Where are datasets loaded?"
→ `utils/dataset_loader.py` → `load_all_datasets()`

### "Where is audio preprocessing?"
→ `utils/audio_processing.py` → `preprocess_audio()`

---

This completes the file overview. Every file serves a specific purpose in the emotion recognition pipeline!
