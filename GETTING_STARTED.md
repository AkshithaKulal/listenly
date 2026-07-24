# 🚀 Getting Started with Speech Emotion Recognition

Welcome! This guide will help you get the Speech Emotion Recognition system up and running quickly.

## ⚡ Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Run the quick start script
python quick_start.py
```

This will check:
- ✅ All required packages
- ✅ Directory structure
- ✅ Module functionality
- ✅ Available datasets

### Step 3: Choose Your Path

You have three options:

**Option A: Use the Web App (Recommended for Beginners)**
```bash
# Windows
run_app.bat

# Linux/Mac
chmod +x run_app.sh
./run_app.sh

# Or directly
streamlit run app.py
```

**Option B: Train Your Own Model**
```bash
python train.py --model cnn_lstm
```

**Option C: Make Predictions via CLI**
```bash
python predict.py path/to/audio.wav
```

---

## 📖 Detailed Walkthrough

### Part 1: Setup Environment

#### Windows
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Linux/Mac
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Part 2: Get Datasets (Optional for Testing)

To train your own model, you need audio datasets. Here's how to get them:

#### Option 1: Download RAVDESS (Recommended)

1. Visit: https://zenodo.org/record/1188976
2. Download "Audio_Speech_Actors_01-24.zip"
3. Extract to `dataset/RAVDESS/`

Final structure:
```
dataset/RAVDESS/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
└── ...
```

#### Option 2: Use Pre-trained Model (Coming Soon)

If you don't want to download datasets, you can use a pre-trained model (requires separate download).

### Part 3: First Training Run

```bash
# Train with default settings (CNN+LSTM model)
python train.py

# This will:
# 1. Load datasets from dataset/ folder
# 2. Extract audio features
# 3. Train the model (may take 20-30 minutes)
# 4. Save the best model to saved_models/
# 5. Generate performance visualizations
```

**Expected Output:**
```
================================================================================
Starting training with model type: cnn_lstm
================================================================================
Loading datasets...
Found 1440 RAVDESS files
Total samples loaded: 1440
Extracting features...
100%|████████████████████| 1440/1440 [05:23<00:00, 4.45it/s]
Train samples: 1152
Validation samples: 144
Test samples: 144
Building CNN+LSTM model...
Starting training...
Epoch 1/50
36/36 [==============================] - 25s 695ms/step - loss: 1.8234 - accuracy: 0.3142 - val_loss: 1.5432 - val_accuracy: 0.4514
...
Training completed!
Test Loss: 0.5234
Test Accuracy: 0.8264
```

### Part 4: Make Your First Prediction

```bash
# Using command line
python predict.py dataset/RAVDESS/Actor_01/03-01-03-01-01-01-01.wav

# Using web app
streamlit run app.py
# Then upload an audio file
```

---

## 🎯 Understanding the Output

### Training Output

After training, you'll find these files in `saved_models/`:

1. **best_model_cnn_lstm.h5** - The trained model (best checkpoint)
2. **final_model_cnn_lstm.h5** - Final model after all epochs
3. **metrics_cnn_lstm.json** - Performance metrics
4. **training_history_cnn_lstm.png** - Training curves
5. **confusion_matrix_cnn_lstm.png** - Confusion matrix
6. **classification_report_cnn_lstm.txt** - Detailed report

### Prediction Output

When you predict an emotion, you'll see:

```
============================================================
🎙️  Emotion Prediction Result
============================================================
Predicted Emotion: HAPPY 😊
Confidence: 87.45%

Probability Distribution:
------------------------------------------------------------
happy      😊 │███████████████████████████████████████████│ 87.45%
neutral    😐 │████                                       │  5.23%
calm       😌 │██                                         │  2.34%
...
============================================================
```

---

## 🎨 Using the Web Application

### Home Page
- Overview of the project
- Supported emotions
- Quick navigation

### Predict Emotion Page
1. **Upload Audio**: Click "Browse files" and select a .wav file
2. **View Results**: See the predicted emotion with confidence
3. **Explore Visualizations**: 
   - Waveform - Audio signal over time
   - Spectrogram - Frequency content
   - MFCC - Feature representation
4. **Download Report**: Get a text report of the prediction

### Model Performance Page
- View training/validation accuracy curves
- Examine confusion matrix
- Read classification report
- Compare model metrics

### About Page
- Technical details
- Architecture information
- Usage instructions

---

## 📊 Testing with Sample Audio

### Creating Test Audio

If you don't have audio files, you can:

1. **Record your own**: Use any recording software to create .wav files
2. **Use online samples**: Search for "emotion speech samples"
3. **Generate synthetic**: Use text-to-speech with emotion

### Audio Requirements

- **Format**: WAV (`.wav`)
- **Sample Rate**: Any (will be resampled to 22050 Hz)
- **Duration**: Any (will be padded/truncated to 3 seconds)
- **Channels**: Mono or Stereo (will be converted to mono)

---

## 🔧 Common Tasks

### Task 1: Train a Different Model

```bash
# Train CNN model
python train.py --model cnn

# Train LSTM model
python train.py --model lstm

# Train CNN+LSTM model (best performance)
python train.py --model cnn_lstm
```

### Task 2: Re-extract Features

```bash
# Force re-extraction (ignore cache)
python train.py --no-cache
```

### Task 3: Batch Predictions

Create a Python script:

```python
from predict import EmotionPredictor

# Initialize predictor
predictor = EmotionPredictor(
    "saved_models/best_model_cnn_lstm.h5",
    model_type="cnn_lstm"
)

# Predict multiple files
files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = predictor.predict_batch(files)

for result in results:
    print(f"{result['file']}: {result['emotion']} ({result['confidence']:.2%})")
```

### Task 4: Adjust Configuration

Edit `config.py`:

```python
# Change audio duration
DURATION = 5.0  # 5 seconds instead of 3

# Increase MFCC coefficients
N_MFCC = 60  # More detail

# Adjust training
BATCH_SIZE = 16  # Smaller batches
EPOCHS = 100     # More epochs
```

---

## 🐛 Troubleshooting

### Problem: "No datasets found"

**Solution:**
1. Download at least one dataset (RAVDESS, TESS, or EMO-DB)
2. Extract to correct folder: `dataset/RAVDESS/`, `dataset/TESS/`, or `dataset/EMO_DB/`
3. Verify structure with `python quick_start.py`

### Problem: "Model not found"

**Solution:**
Train a model first: `python train.py --model cnn_lstm`

### Problem: "Out of memory"

**Solution:**
1. Reduce batch size in `config.py`: `BATCH_SIZE = 16`
2. Use feature caching (enabled by default)
3. Close other applications

### Problem: "Audio format not supported"

**Solution:**
Convert to WAV format:
```bash
# Using ffmpeg
ffmpeg -i input.mp3 output.wav
```

### Problem: "Streamlit not opening"

**Solution:**
```bash
# Try with explicit port
streamlit run app.py --server.port 8501

# Or use Python module
python -m streamlit run app.py
```

---

## 📚 Next Steps

Once you're comfortable with the basics:

1. **📖 Read Full Documentation**: Check `README.md` for detailed information
2. **🔬 Experiment**: Try different models and hyperparameters
3. **📊 Analyze Results**: Study confusion matrices to understand model behavior
4. **🎨 Customize UI**: Modify `app.py` to add your own features
5. **📈 Improve Model**: Try data augmentation or ensemble methods

---

## 💡 Tips for Best Results

### Training Tips
- Use multiple datasets for better generalization
- Enable data augmentation for more samples
- Monitor validation loss to avoid overfitting
- Try different model architectures

### Prediction Tips
- Use high-quality audio recordings
- Minimize background noise
- Speak clearly with emotion
- Test with diverse speakers

### Performance Tips
- Use GPU for faster training (if available)
- Enable feature caching to speed up training
- Use smaller batch sizes if memory is limited

---

## 🎓 Learning Resources

### Understanding Audio Processing
- Librosa documentation: https://librosa.org
- Audio signal processing basics
- MFCC feature explanation

### Deep Learning Concepts
- TensorFlow tutorials: https://tensorflow.org/tutorials
- CNN architectures
- RNN/LSTM networks

### Streamlit Development
- Streamlit documentation: https://docs.streamlit.io
- Building interactive UIs
- Session state management

---

## 📞 Getting Help

If you're stuck:

1. **Check Documentation**:
   - README.md - Main guide
   - INSTALL.md - Installation issues
   - PROJECT_SUMMARY.md - Technical overview

2. **Run Diagnostics**:
   ```bash
   python quick_start.py
   ```

3. **Check Logs**:
   - Training logs: `ser_training.log`
   - Error messages in console

4. **Common Issues**:
   - Review troubleshooting section above
   - Check requirements.txt versions

---

## ✅ Checklist for Success

- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Quick start test passed (`python quick_start.py`)
- [ ] At least one dataset downloaded (optional for testing)
- [ ] Model trained or pre-trained model available
- [ ] Web app runs successfully (`streamlit run app.py`)
- [ ] Predictions work correctly

---

## 🎉 You're Ready!

Congratulations! You're now ready to use the Speech Emotion Recognition system.

**What's Next?**
- 🎯 Try predicting emotions from your own voice
- 📊 Train models on different datasets
- 🎨 Customize the web interface
- 📈 Analyze model performance
- 🚀 Deploy to production

**Have fun exploring emotions in speech! 🎙️😊**

---

For more information, see:
- 📖 README.md - Complete documentation
- 🔧 INSTALL.md - Detailed installation
- 📊 PROJECT_SUMMARY.md - Technical details

*Happy emotion recognition!* 🎭
