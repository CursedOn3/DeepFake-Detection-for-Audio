# DeepFake Audio Detection System

🎙️ **Audio-Only adaptation of the ERF-BA-TFD+ architecture for deepfake detection**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ⚠️ IMPORTANT: Read This First!

**This system needs to be TRAINED before it can detect deepfakes!**

📖 **Quick Links:**
- 👉 **[START_HERE.md](START_HERE.md)** - Everything you need to know (READ THIS!)
- 📋 **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup instructions
- 📁 **Project Structure** - See below

---

## 🚀 Quick Start (After Training)

```python
from main import detect_deepfake

# Single function call!
result = detect_deepfake(
    "path/to/audio.wav",
    model_path="outputs/checkpoint_best.pth"  # Your trained model
)

print(result['label'])      # "REAL" or "FAKE"
print(result['fake_score']) # 0.0 (real) to 1.0 (fake)
```

## 📦 Installation & Training

```bash
# 1. Install dependencies
pip install -r requirements.txt
python verify_installation.py

# 2. Organize dataset
# data/train/real/ ← real audio files
# data/train/fake/ ← fake audio files
# data/val/real/   ← validation real
# data/val/fake/   ← validation fake

# 3. Train the model
python train.py --data_dir data/ --batch_size 16 --num_epochs 50

# 4. Test detection
python main.py --audio test.wav --model outputs/checkpoint_best.pth
```

---

## Overview

This system detects AI-generated (deepfake) audio using a Transformer-based architecture that combines:
- **Audio Encoding**: CNN + Transformer to capture temporal patterns
- **Anomaly Features**: Spectral and prosody features to detect synthesis artifacts
- **Feature Fusion**: Combines both branches for robust detection