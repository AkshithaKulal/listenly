"""
quick_start.py - Quick demonstration script
Tests the installation and shows basic usage.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_imports():
    """Check if all required packages are installed."""
    print_header("Checking Required Packages")
    
    required = {
        'tensorflow': 'TensorFlow',
        'librosa': 'Librosa',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'streamlit': 'Streamlit',
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name:20s} - OK")
        except ImportError:
            print(f"❌ {name:20s} - NOT FOUND")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check if required directories exist."""
    print_header("Checking Directory Structure")
    
    import config
    
    dirs = {
        'Dataset': config.DATASET_DIR,
        'Saved Models': config.SAVED_MODELS_DIR,
        'Models': config.MODELS_DIR,
        'Utils': os.path.join(config.BASE_DIR, 'utils'),
    }
    
    for name, path in dirs.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "⚠️"
        print(f"{status} {name:20s} - {path}")
        
        if not exists and name in ['Saved Models', 'Models']:
            os.makedirs(path, exist_ok=True)
            print(f"   Created {path}")


def check_datasets():
    """Check if datasets are available."""
    print_header("Checking Datasets")
    
    import config
    import glob
    
    datasets = {
        'RAVDESS': config.RAVDESS_DIR,
        'TESS': config.TESS_DIR,
        'EMO-DB': config.EMO_DB_DIR,
    }
    
    total_files = 0
    
    for name, path in datasets.items():
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
            count = len(files)
            total_files += count
            if count > 0:
                print(f"✅ {name:10s} - {count:5d} audio files found")
            else:
                print(f"⚠️  {name:10s} - Directory exists but no files found")
        else:
            print(f"❌ {name:10s} - Not found at {path}")
    
    print(f"\nTotal audio files: {total_files}")
    
    if total_files == 0:
        print("\n⚠️  WARNING: No audio files found!")
        print("   Please download and extract datasets to the dataset/ directory")
        print("   See README.md for dataset download instructions")
    
    return total_files > 0


def test_audio_processing():
    """Test audio processing functions."""
    print_header("Testing Audio Processing")
    
    try:
        from utils.audio_processing import preprocess_audio, normalize_audio
        import numpy as np
        
        # Create dummy audio
        dummy_audio = np.random.randn(22050 * 2)  # 2 seconds
        
        # Test normalization
        normalized = normalize_audio(dummy_audio)
        print("✅ Audio normalization - OK")
        
        print("✅ Audio processing module - OK")
        return True
        
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction functions."""
    print_header("Testing Feature Extraction")
    
    try:
        from utils.feature_extraction import extract_mfcc
        import numpy as np
        
        # Create dummy audio
        dummy_audio = np.random.randn(22050 * 2)
        sr = 22050
        
        # Extract MFCC
        mfcc = extract_mfcc(dummy_audio, sr, n_mfcc=40)
        print(f"✅ MFCC extraction - OK (shape: {mfcc.shape})")
        
        print("✅ Feature extraction module - OK")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False


def test_model_building():
    """Test model building."""
    print_header("Testing Model Building")
    
    try:
        from models import build_cnn_lstm_model
        import tensorflow as tf
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        
        # Build a small model
        model = build_cnn_lstm_model(
            input_shape=(40, 100, 1),
            num_classes=8,
            learning_rate=0.001
        )
        
        print(f"✅ Model built successfully")
        print(f"   Total parameters: {model.count_params():,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model building test failed: {e}")
        return False


def show_next_steps(has_datasets):
    """Show next steps to user."""
    print_header("Next Steps")
    
    if not has_datasets:
        print("\n1. 📥 DOWNLOAD DATASETS")
        print("   Download RAVDESS, TESS, or EMO-DB datasets")
        print("   See README.md for download links")
        print("   Extract to dataset/ directory")
        print()
    
    print("2. 🎓 TRAIN A MODEL")
    print("   python train.py --model cnn_lstm")
    print()
    
    print("3. 🔮 MAKE PREDICTIONS")
    print("   python predict.py path/to/audio.wav")
    print()
    
    print("4. 🌐 LAUNCH WEB APP")
    print("   streamlit run app.py")
    print()
    
    print("5. 📖 READ DOCUMENTATION")
    print("   See README.md for detailed instructions")
    print("   See INSTALL.md for installation help")
    print()


def main():
    """Main quick start function."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                                                                  ║")
    print("║        🎙️  SPEECH EMOTION RECOGNITION - QUICK START 🎙️         ║")
    print("║                                                                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Run checks
    imports_ok = check_imports()
    
    if not imports_ok:
        print("\n❌ Some packages are missing!")
        print("   Run: pip install -r requirements.txt")
        return 1
    
    check_directories()
    has_datasets = check_datasets()
    
    # Run tests
    audio_ok = test_audio_processing()
    features_ok = test_feature_extraction()
    model_ok = test_model_building()
    
    # Summary
    print_header("Summary")
    
    all_tests = [imports_ok, audio_ok, features_ok, model_ok]
    passed = sum(all_tests)
    total = len(all_tests)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All systems operational!")
        print("✅ Installation verified successfully!")
    else:
        print("\n⚠️  Some tests failed. Please check error messages above.")
    
    # Next steps
    show_next_steps(has_datasets)
    
    print_header("Quick Start Complete")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
