# Installation Guide

## Prerequisites

Before installing, ensure you have:
- Python 3.11 or higher
- pip package manager
- At least 4GB of RAM
- (Optional) NVIDIA GPU with CUDA for faster training

## Step-by-Step Installation

### 1. System Setup

#### Windows
```bash
# Check Python version
python --version

# Should show Python 3.11.x or higher
```

#### Linux/Mac
```bash
# Check Python version
python3 --version

# Install pip if not available
sudo apt-get install python3-pip  # Ubuntu/Debian
brew install python3               # Mac with Homebrew
```

### 2. Project Setup

```bash
# Navigate to project directory
cd SpeechEmotionRecognition

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This will install:
- TensorFlow (Deep Learning)
- Librosa (Audio Processing)
- Streamlit (Web Interface)
- NumPy, Pandas (Data Processing)
- Matplotlib, Seaborn (Visualization)
- And more...

### 4. Verify Installation

```bash
# Test imports
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import librosa; print('Librosa:', librosa.__version__)"
python -c "import streamlit as st; print('Streamlit:', st.__version__)"
```

Expected output:
```
TensorFlow: 2.13.0 (or higher)
Librosa: 0.10.0 (or higher)
Streamlit: 1.28.0 (or higher)
```

### 5. Download Datasets

#### Option A: Manual Download

**RAVDESS Dataset**
1. Visit: https://zenodo.org/record/1188976
2. Download the audio files
3. Extract to: `dataset/RAVDESS/`

**TESS Dataset**
1. Visit: https://tspace.library.utoronto.ca/handle/1807/24487
2. Download the dataset
3. Extract to: `dataset/TESS/`

**EMO-DB Dataset**
1. Visit: http://emodb.bilderbar.info/download/
2. Download the wav files
3. Extract to: `dataset/EMO_DB/`

#### Option B: Using Scripts (if available)

```bash
# Download all datasets (if script provided)
python download_datasets.py
```

### 6. Verify Dataset Structure

```bash
# Check dataset structure
python -c "import os; from config import *; print('RAVDESS exists:', os.path.exists(RAVDESS_DIR)); print('TESS exists:', os.path.exists(TESS_DIR)); print('EMO-DB exists:', os.path.exists(EMO_DB_DIR))"
```

Expected structure:
```
dataset/
├── RAVDESS/
│   ├── Actor_01/
│   │   └── *.wav files
│   └── ...
├── TESS/
│   └── *.wav files
└── EMO_DB/
    └── *.wav files
```

### 7. First Run

```bash
# Test training (will process a small sample)
python train.py --model cnn_lstm

# If successful, the model will be saved to saved_models/
```

### 8. Launch Web App

```bash
# Start Streamlit app
streamlit run app.py

# App will open at http://localhost:8501
```

## Common Installation Issues

### Issue 1: TensorFlow Installation Failed

**Windows:**
```bash
# Install Microsoft Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe

# Then retry
pip install tensorflow
```

**Mac M1/M2:**
```bash
# Install TensorFlow for Apple Silicon
pip install tensorflow-macos
pip install tensorflow-metal
```

### Issue 2: Librosa Audio Backend Error

```bash
# Install audio backend
pip install soundfile
# OR
pip install audioread

# For Linux, may need system libraries
sudo apt-get install libsndfile1
```

### Issue 3: Memory Error During Training

**Solution:**
Edit `config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### Issue 4: CUDA/GPU Not Detected

```bash
# Verify GPU
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

# Install CUDA toolkit if needed
# Visit: https://developer.nvidia.com/cuda-downloads
```

### Issue 5: Streamlit Command Not Found

```bash
# Ensure virtual environment is activated
# Then try full path
python -m streamlit run app.py
```

## GPU Setup (Optional, for Faster Training)

### NVIDIA GPU Setup

1. **Check GPU Compatibility**
   ```bash
   nvidia-smi
   ```

2. **Install CUDA Toolkit**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Install CUDA 11.8 or 12.x

3. **Install cuDNN**
   - Download from: https://developer.nvidia.com/cudnn
   - Extract and add to PATH

4. **Install TensorFlow GPU**
   ```bash
   pip install tensorflow[and-cuda]
   ```

5. **Verify GPU**
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

## Testing Installation

Run the test suite:

```bash
# Test audio processing
python -c "from utils.audio_processing import preprocess_audio; print('Audio processing: OK')"

# Test feature extraction
python -c "from utils.feature_extraction import extract_mfcc; print('Feature extraction: OK')"

# Test model building
python -c "from models import build_cnn_lstm_model; print('Model building: OK')"
```

## Uninstallation

To completely remove the environment:

```bash
# Deactivate virtual environment
deactivate

# Delete virtual environment
# Windows:
rmdir /s venv
# Linux/Mac:
rm -rf venv

# Remove cached files
# Windows:
del /s /q __pycache__
# Linux/Mac:
find . -type d -name __pycache__ -exec rm -rf {} +
```

## Next Steps

After successful installation:

1. **Train a Model**
   ```bash
   python train.py --model cnn_lstm
   ```

2. **Test Prediction**
   ```bash
   python predict.py dataset/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav
   ```

3. **Launch Web App**
   ```bash
   streamlit run app.py
   ```

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Review error messages carefully
3. Search for similar issues online
4. Check library documentation:
   - TensorFlow: https://tensorflow.org
   - Librosa: https://librosa.org
   - Streamlit: https://streamlit.io

## Minimum System Requirements

- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **CPU**: Intel i5 or equivalent
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space (more for datasets)
- **GPU**: Optional, but recommended for training

## Recommended System Specifications

- **CPU**: Intel i7/AMD Ryzen 7 or better
- **RAM**: 16GB or more
- **GPU**: NVIDIA GTX 1060 or better (6GB+ VRAM)
- **Storage**: 20GB free space (SSD preferred)

---

**Installation Complete! 🎉**

You're now ready to build and use the Speech Emotion Recognition system.
