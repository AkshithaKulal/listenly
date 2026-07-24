"""
app.py - Streamlit Web Application for Speech Emotion Recognition
Professional UI with multiple pages and interactive features.
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import librosa
import soundfile as sf
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from predict import EmotionPredictor
from utils.visualization import (
    plot_waveform,
    plot_spectrogram,
    plot_mfcc,
    plot_prediction_probabilities
)
from utils.audio_processing import preprocess_audio
from utils.feature_extraction import extract_mfcc

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6C63FF;
        --secondary-color: #4CAF50;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Cards */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #6C63FF;
    }
    
    /* Emotion result */
    .emotion-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .confidence-score {
        font-size: 1.5rem;
        color: #4CAF50;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #6C63FF;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────

if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

@st.cache_resource
def load_predictor(model_type="cnn_lstm"):
    """Load the emotion predictor model."""
    try:
        model_path = os.path.join(config.SAVED_MODELS_DIR, f"best_model_{model_type}.h5")
        if not os.path.exists(model_path):
            model_path = os.path.join(config.SAVED_MODELS_DIR, f"final_model_{model_type}.h5")
        
        if not os.path.exists(model_path):
            return None, f"Model not found: {model_path}"
        
        predictor = EmotionPredictor(model_path, model_type=model_type)
        return predictor, None
    except Exception as e:
        return None, str(e)


def add_to_history(filename, emotion, confidence):
    """Add prediction to history."""
    st.session_state.prediction_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'filename': filename,
        'emotion': emotion,
        'confidence': confidence
    })
    # Keep only last N entries
    if len(st.session_state.prediction_history) > config.HISTORY_SIZE:
        st.session_state.prediction_history.pop(0)


def create_download_report(result, filename):
    """Create a downloadable prediction report."""
    report = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║        SPEECH EMOTION RECOGNITION REPORT                ║
    ╚══════════════════════════════════════════════════════════╝
    
    File: {filename}
    Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    ──────────────────────────────────────────────────────────
    PREDICTION RESULT
    ──────────────────────────────────────────────────────────
    
    Emotion: {result['emotion'].upper()}
    Confidence: {result['confidence']:.2%}
    
    ──────────────────────────────────────────────────────────
    PROBABILITY DISTRIBUTION
    ──────────────────────────────────────────────────────────
    
    """
    
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_probs:
        bar = "█" * int(prob * 40)
        report += f"\n    {emotion:10s} │{bar:<40}│ {prob:6.2%}"
    
    report += "\n\n    ──────────────────────────────────────────────────────────\n"
    report += "    Generated by Speech Emotion Recognition System\n"
    report += "    ──────────────────────────────────────────────────────────\n"
    
    return report


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: white;'>🎙️ SER</h1>
        <p style='color: white; opacity: 0.9;'>Speech Emotion Recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🎯 Predict Emotion", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model selection
    st.subheader("⚙️ Settings")
    model_type = st.selectbox(
        "Model Type",
        ["cnn_lstm", "cnn", "lstm"],
        index=0,
        help="Choose the model architecture"
    )
    
    # Dark mode toggle
    dark_mode = st.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.markdown("---")
    
    # Load model
    if st.session_state.predictor is None or st.button("🔄 Reload Model"):
        with st.spinner("Loading model..."):
            predictor, error = load_predictor(model_type)
            if error:
                st.error(f"Error loading model: {error}")
            else:
                st.session_state.predictor = predictor
                st.success("✅ Model loaded successfully!")
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")


# ─────────────────────────────────────────────
# Page: Home
# ─────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown("""
    <div class='main-header'>
        <h1>🎙️ Speech Emotion Recognition</h1>
        <p>Recognize emotions from speech using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>🎯 Accurate</h3>
            <p>State-of-the-art deep learning models achieve high accuracy in emotion classification.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>⚡ Fast</h3>
            <p>Real-time prediction with optimized neural network architectures.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h3>🔧 Easy to Use</h3>
            <p>Simple upload and instant emotion analysis with detailed visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. Go to **🎯 Predict Emotion** page
    2. Upload a `.wav` audio file or record your voice
    3. Click **Predict Emotion**
    4. View results and visualizations
    """)
    
    st.markdown("### 😊 Supported Emotions")
    
    cols = st.columns(4)
    emotions_display = [
        ("😐 Neutral", "neutral"),
        ("😌 Calm", "calm"),
        ("😊 Happy", "happy"),
        ("😢 Sad", "sad"),
        ("😠 Angry", "angry"),
        ("😨 Fear", "fear"),
        ("🤢 Disgust", "disgust"),
        ("😲 Surprise", "surprise"),
    ]
    
    for idx, (emoji_label, _) in enumerate(emotions_display):
        with cols[idx % 4]:
            st.markdown(f"### {emoji_label}")
    
    st.markdown("### 📈 Model Architecture")
    st.markdown("""
    The system uses a hybrid **CNN + LSTM** architecture:
    - **CNN layers**: Extract spatial features from MFCC spectrograms
    - **LSTM layers**: Capture temporal patterns in speech
    - **Dense layers**: Final classification into emotion categories
    """)
    
    st.markdown("### 🎵 Supported Datasets")
    st.markdown("""
    - **RAVDESS**: Ryerson Audio-Visual Database of Emotional Speech and Song
    - **TESS**: Toronto Emotional Speech Set
    - **EMO-DB**: Berlin Database of Emotional Speech
    """)


# ─────────────────────────────────────────────
# Page: Predict Emotion
# ─────────────────────────────────────────────

elif page == "🎯 Predict Emotion":
    st.markdown("""
    <div class='main-header'>
        <h1>🎯 Predict Emotion from Speech</h1>
        <p>Upload an audio file or record your voice</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.predictor is None:
        st.warning("⚠️ Model not loaded. Please load the model from the sidebar.")
        st.stop()
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload Audio File", "🎤 Record Audio"],
        horizontal=True
    )
    
    audio_file = None
    audio_data = None
    filename = None
    
    if input_method == "📁 Upload Audio File":
        uploaded_file = st.file_uploader(
            "Upload a .wav file",
            type=["wav"],
            help="Maximum file size: 200MB"
        )
        
        if uploaded_file is not None:
            audio_file = uploaded_file
            filename = uploaded_file.name
            audio_data = uploaded_file.read()
            
            st.audio(audio_data, format='audio/wav')
            st.success(f"✅ File uploaded: {filename}")
    
    else:  # Record Audio
        st.info("🎤 Audio recording feature requires browser permissions.")
        st.markdown("**Note**: Recording functionality requires additional setup. Please upload a file instead.")
    
    # Prediction section
    if audio_file is not None:
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Options")
            show_waveform = st.checkbox("Show Waveform", value=True)
            show_spectrogram = st.checkbox("Show Spectrogram", value=True)
            show_mfcc = st.checkbox("Show MFCC", value=True)
            
            predict_button = st.button("🚀 Predict Emotion", use_container_width=True, type="primary")
        
        if predict_button:
            with st.spinner("🔍 Analyzing audio..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_file.read() if hasattr(audio_file, 'read') else audio_data)
                        tmp_path = tmp_file.name
                    
                    # Make prediction
                    result = st.session_state.predictor.predict(tmp_path)
                    
                    # Display result
                    emotion = result['emotion']
                    confidence = result['confidence']
                    emoji = config.EMOTION_EMOJI.get(emotion, "")
                    
                    st.markdown(f"""
                    <div class='emotion-result'>
                        <div style='font-size: 3rem;'>{emoji}</div>
                        <div>Emotion: {emotion.upper()}</div>
                        <div class='confidence-score'>Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add to history
                    add_to_history(filename, emotion, confidence)
                    
                    # Probability distribution
                    st.markdown("### 📊 Probability Distribution")
                    
                    # Create bar chart
                    probs_array = np.array([result['probabilities'][label] for label in config.EMOTION_LABELS])
                    fig_probs = plot_prediction_probabilities(probs_array, config.EMOTION_LABELS)
                    st.pyplot(fig_probs)
                    plt.close(fig_probs)
                    
                    # Detailed probabilities table
                    with st.expander("📋 Detailed Probabilities"):
                        prob_df = pd.DataFrame([
                            {
                                'Emotion': emotion.capitalize(),
                                'Emoji': config.EMOTION_EMOJI.get(emotion, ""),
                                'Probability': f"{prob:.2%}",
                                'Value': prob
                            }
                            for emotion, prob in sorted(
                                result['probabilities'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                        ])
                        st.dataframe(prob_df[['Emoji', 'Emotion', 'Probability']], use_container_width=True, hide_index=True)
                    
                    # Visualizations
                    if show_waveform or show_spectrogram or show_mfcc:
                        st.markdown("### 🎵 Audio Visualizations")
                        
                        # Load audio for visualization
                        audio, sr = librosa.load(tmp_path, sr=config.SAMPLE_RATE)
                        
                        if show_waveform:
                            with st.expander("🌊 Waveform", expanded=True):
                                fig_wave = plot_waveform(audio, sr)
                                st.pyplot(fig_wave)
                                plt.close(fig_wave)
                        
                        if show_spectrogram:
                            with st.expander("📈 Spectrogram", expanded=True):
                                fig_spec = plot_spectrogram(audio, sr)
                                st.pyplot(fig_spec)
                                plt.close(fig_spec)
                        
                        if show_mfcc:
                            with st.expander("🔥 MFCC Heatmap", expanded=True):
                                mfcc = extract_mfcc(audio, sr, n_mfcc=config.N_MFCC)
                                fig_mfcc = plot_mfcc(mfcc, sr)
                                st.pyplot(fig_mfcc)
                                plt.close(fig_mfcc)
                    
                    # Download report
                    st.markdown("### 💾 Download Report")
                    report_text = create_download_report(result, filename)
                    st.download_button(
                        label="📄 Download Prediction Report",
                        data=report_text,
                        file_name=f"emotion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
        
        # Prediction History
        if st.session_state.prediction_history:
            st.markdown("---")
            st.markdown("### 📜 Prediction History")
            
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['confidence'] = history_df['confidence'].apply(lambda x: f"{x:.1%}")
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()


# ─────────────────────────────────────────────
# Page: Model Performance
# ─────────────────────────────────────────────

elif page == "📊 Model Performance":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Model Performance</h1>
        <p>Training metrics and evaluation results</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load metrics
    model_type_display = st.selectbox("Select Model", ["cnn_lstm", "cnn", "lstm"])
    
    metrics_path = os.path.join(config.SAVED_MODELS_DIR, f"metrics_{model_type_display}.json")
    
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-value'>{metrics['test_accuracy']:.1%}</div>
                <div class='metric-label'>Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-value'>{metrics['test_loss']:.3f}</div>
                <div class='metric-label'>Test Loss</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-value'>{metrics['epochs_trained']}</div>
                <div class='metric-label'>Epochs Trained</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class='metric-container'>
                <div class='metric-value'>{metrics['train_samples']}</div>
                <div class='metric-label'>Training Samples</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display plots
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 📈 Training History")
            history_img = os.path.join(config.SAVED_MODELS_DIR, f"training_history_{model_type_display}.png")
            if os.path.exists(history_img):
                st.image(history_img, use_container_width=True)
            else:
                st.info("Training history plot not available")
        
        with col_right:
            st.markdown("### 🎯 Confusion Matrix")
            cm_img = os.path.join(config.SAVED_MODELS_DIR, f"confusion_matrix_{model_type_display}.png")
            if os.path.exists(cm_img):
                st.image(cm_img, use_container_width=True)
            else:
                st.info("Confusion matrix not available")
        
        # Classification report
        st.markdown("### 📋 Classification Report")
        report_path = os.path.join(config.SAVED_MODELS_DIR, f"classification_report_{model_type_display}.txt")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report_text = f.read()
            st.code(report_text, language='text')
        else:
            st.info("Classification report not available")
    
    else:
        st.warning(f"⚠️ No metrics found for {model_type_display} model. Please train the model first.")
        st.markdown("""
        To train a model, run:
        ```bash
        python train.py --model cnn_lstm
        ```
        """)


# ─────────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────────

elif page == "ℹ️ About":
    st.markdown("""
    <div class='main-header'>
        <h1>ℹ️ About This Project</h1>
        <p>Speech Emotion Recognition using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This Speech Emotion Recognition (SER) system uses advanced deep learning techniques to analyze 
    and classify emotions from speech audio. The system can identify 8 different emotions:
    Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, and Surprise.
    
    ### 🔬 Technical Details
    
    **Architecture**: Hybrid CNN + LSTM
    - **CNN Layers**: Extract spatial features from MFCC spectrograms
    - **LSTM Layers**: Capture temporal dependencies in speech patterns
    - **Dense Layers**: Final emotion classification
    
    **Features Extracted**:
    - MFCC (Mel-Frequency Cepstral Coefficients)
    - Mel Spectrogram
    - Chroma Features
    - Zero Crossing Rate
    - RMS Energy
    - Spectral Centroid
    
    **Training Datasets**:
    - RAVDESS (Ryerson Audio-Visual Database)
    - TESS (Toronto Emotional Speech Set)
    - EMO-DB (Berlin Emotional Speech Database)
    
    ### 🛠️ Technology Stack
    
    - **Python 3.11+**
    - **TensorFlow/Keras**: Deep learning framework
    - **Librosa**: Audio processing and feature extraction
    - **Streamlit**: Interactive web interface
    - **NumPy & Pandas**: Data manipulation
    - **Matplotlib & Seaborn**: Visualizations
    - **Scikit-learn**: Model evaluation
    
    ### 📊 Model Performance
    
    The CNN+LSTM hybrid model achieves high accuracy on test datasets, with detailed 
    performance metrics available in the Model Performance page.
    
    ### 👨‍💻 Usage
    
    **Training a Model**:
    ```bash
    python train.py --model cnn_lstm
    ```
    
    **Making Predictions**:
    ```bash
    python predict.py audio_file.wav
    ```
    
    **Running the Web App**:
    ```bash
    streamlit run app.py
    ```
    
    ### 📝 Project Structure
    
    ```
    SpeechEmotionRecognition/
    ├── dataset/           # Audio datasets
    ├── models/            # Model architectures
    ├── saved_models/      # Trained models and metrics
    ├── utils/             # Utility functions
    ├── train.py           # Training script
    ├── predict.py         # Prediction script
    ├── app.py             # Streamlit web app
    └── config.py          # Configuration settings
    ```
    
    ### 🎓 Applications
    
    - **Customer Service**: Analyze customer emotions in call centers
    - **Mental Health**: Monitor emotional states in therapy
    - **Human-Computer Interaction**: Create emotionally aware systems
    - **Entertainment**: Enhance gaming and VR experiences
    - **Education**: Assess student engagement and emotions
    
    ### 📄 License
    
    This project is created for educational and research purposes.
    
    ### 🙏 Acknowledgments
    
    - RAVDESS, TESS, and EMO-DB dataset creators
    - TensorFlow and Keras teams
    - Open-source community
    
    ---
    
    **Version**: 1.0.0  
    **Last Updated**: 2024
    
    Made with ❤️ for final year engineering project
    """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.6; padding: 1rem;'>
    <p>🎙️ Speech Emotion Recognition System | Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
