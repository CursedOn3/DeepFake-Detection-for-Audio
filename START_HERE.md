# 🎯 QUICK START - EVERYTHING YOU NEED TO KNOW

## Current Status: ⚠️ MODEL NOT TRAINED YET

Your deepfake detection system is **CODE-COMPLETE** but needs **TRAINING** before it can detect anything!

---

## 🔥 Critical Steps to Make It Work

### Step 1: Install Everything (5 minutes)

```bash
# Install all packages
pip install -r requirements.txt

# Check installation
python verify_installation.py
```

### Step 2: Get Audio Data (30-60 minutes)

You **MUST** have audio samples to train on!

**Option A: Download Public Datasets**
- ASVspoof: https://www.asvspoof.org/
- WaveFake: https://github.com/RUB-SysSec/WaveFake
- FakeAVCeleb: https://github.com/DASH-Lab/FakeAVCeleb

**Option B: Create Your Own**
- Record real human speech (100+ samples)
- Generate fake speech using TTS tools like:
  - ElevenLabs
  - Coqui TTS
  - Tortoise TTS

### Step 3: Organize Data (10 minutes)

```
data/
├── train/
│   ├── real/       ← Put 80% of REAL audio here
│   │   ├── real1.wav
│   │   ├── real2.wav
│   │   └── ... (500+ files recommended)
│   └── fake/       ← Put 80% of FAKE audio here
│       ├── fake1.wav
│       ├── fake2.wav
│       └── ... (500+ files recommended)
└── val/
    ├── real/       ← Put 20% of REAL audio here
    │   └── ... (100+ files)
    └── fake/       ← Put 20% of FAKE audio here
        └── ... (100+ files)
```

### Step 4: Train the Model (2-4 hours on GPU)

```bash
python train.py --data_dir data/ --batch_size 16 --num_epochs 50
```

**Watch training progress:**
```bash
# In another terminal
tensorboard --logdir outputs/logs/
# Open http://localhost:6006 in browser
```

### Step 5: Test It! (1 minute)

```bash
python main.py --audio test_audio.wav --model outputs/checkpoint_best.pth
```

---

## 📊 What Each File Does

| File | Purpose | Status |
|------|---------|--------|
| `model/audio_encoder.py` | Transformer model to understand audio | ✅ READY |
| `model/anomaly_extractor.py` | Extracts suspicious features | ✅ READY |
| `model/classifier.py` | Decides real vs fake | ✅ READY |
| `data_loader.py` | Loads audio files for training | ✅ READY |
| `train.py` | Trains the model | ✅ READY |
| `main.py` | Detects deepfakes | ✅ READY |
| `configs/config.yaml` | All settings | ✅ READY |
| `verify_installation.py` | Checks dependencies | ✅ READY |

---

## ⚙️ How the Detection Works

```
Audio File (test.wav)
    ↓
1. Load & Preprocess (16kHz, normalize)
    ↓
2. Create Mel-Spectrogram (visual representation)
    ↓
3. Audio Encoder (Transformer analyzes patterns)
    ↓
4. Anomaly Extractor (finds suspicious features)
    ↓
5. Classifier (combines info, outputs score)
    ↓
Fake Score: 0.0-1.0
  • 0.0 = REAL
  • 1.0 = FAKE
  • 0.5 = UNCERTAIN
```

---

## 🎓 Training Explained

**What is training?**
- The model starts with random weights (knows nothing)
- It looks at examples of real and fake audio
- It learns patterns that distinguish them
- After training, it can detect fakes it's never seen

**What you need:**
- Minimum: 500 real + 500 fake audio files
- Recommended: 2000+ of each for good accuracy
- Best: 10,000+ of each for high accuracy

**How long does it take?**
- With GPU: 2-4 hours (50 epochs)
- Without GPU: 20-40 hours (50 epochs)

---

## 🚨 Common Mistakes

### ❌ Mistake 1: Running without training
```bash
# This WON'T work (no trained model)
python main.py --audio test.wav
```
**Fix:** Train first, then specify the model:
```bash
python main.py --audio test.wav --model outputs/checkpoint_best.pth
```

### ❌ Mistake 2: Not enough data
```
data/train/real/  ← Only 10 files (TOO FEW!)
data/train/fake/  ← Only 10 files (TOO FEW!)
```
**Fix:** Need at least 500+ files per class

### ❌ Mistake 3: Wrong folder structure
```
data/
  ├── audio1.wav  ← WRONG!
  └── audio2.wav  ← WRONG!
```
**Fix:** Must use train/real/ and train/fake/ folders

### ❌ Mistake 4: CUDA out of memory
```bash
# Batch size too large
python train.py --batch_size 64  ← CRASHES!
```
**Fix:** Use smaller batch size:
```bash
python train.py --batch_size 8
```

---

## 📈 What to Expect

### Training Progress (typical)
```
Epoch 1/50:  Loss: 0.693, Acc: 52%  ← Random guessing
Epoch 10/50: Loss: 0.421, Acc: 78%  ← Learning!
Epoch 30/50: Loss: 0.152, Acc: 94%  ← Good!
Epoch 50/50: Loss: 0.089, Acc: 97%  ← Excellent!
```

### Good Signs ✅
- Loss decreases over time
- Accuracy increases over time
- Validation accuracy stays close to training accuracy

### Bad Signs ⚠️
- Loss stays around 0.69 (model not learning)
- Accuracy stays at 50% (random guessing)
- Training accuracy = 99%, validation = 60% (overfitting)

---

## 🎯 Testing Your Model

After training, test on new audio:

```python
from main import detect_deepfake

# Single file
result = detect_deepfake("test.wav", model_path="outputs/checkpoint_best.pth")
print(f"Label: {result['label']}")         # REAL or FAKE
print(f"Score: {result['fake_score']:.2f}") # 0.0 to 1.0
print(f"Confidence: {result['confidence']:.0%}") # How sure it is

# Multiple files
from main import SimpleDeepfakeDetector

detector = SimpleDeepfakeDetector("outputs/checkpoint_best.pth")
results = detector.check_multiple_files([
    "test1.wav",
    "test2.wav",
    "test3.wav"
])

for r in results:
    print(f"{r['file']}: {r['label']} ({r['fake_score']:.2f})")
```

---

## 💡 Tips for Best Results

1. **Balanced Dataset**: Equal number of real and fake samples
2. **Diverse Data**: Different speakers, accents, recording conditions
3. **Quality Audio**: Use WAV or FLAC (not low-quality MP3)
4. **Multiple Fake Types**: Include various deepfake methods (TTS, voice cloning, etc.)
5. **Clean Labels**: Double-check that real is really real, fake is really fake
6. **Sufficient Length**: Audio clips of 2-10 seconds work best
7. **Monitor Training**: Use TensorBoard to watch progress
8. **Experiment**: Try different hyperparameters if results aren't good

---

## 🆘 Emergency Troubleshooting

**Model gives same prediction for everything:**
- Needs more training or better data
- Check that dataset has both real and fake samples
- Verify labels are correct

**Training crashes:**
- Reduce batch size: `--batch_size 4`
- Check if you're running out of RAM/GPU memory
- Update PyTorch: `pip install --upgrade torch`

**Low accuracy after training:**
- Need more training data (>1000 samples per class)
- Try training longer (100+ epochs)
- Data might be too similar (add more diversity)
- Check if audio quality is consistent

**Can't find trained model:**
- Model is saved in `outputs/checkpoint_best.pth`
- Must complete training before this file exists
- If training crashed, it won't be created

---

## ✅ Final Checklist

Before asking "Why doesn't it work?", verify:

- [ ] All packages installed (`python verify_installation.py`)
- [ ] Dataset organized in correct folders (data/train/real/, data/train/fake/, etc.)
- [ ] At least 500+ audio files per class (real and fake)
- [ ] Training completed successfully (no crashes)
- [ ] Model file exists (`outputs/checkpoint_best.pth`)
- [ ] Using correct model path when testing (`--model outputs/checkpoint_best.pth`)

---

## 🚀 You're Ready!

Your system is **complete** - now you just need to:
1. ✅ Install dependencies
2. 📁 Get/organize audio data
3. 🎯 Train the model
4. 🔍 Test it!

Good luck! 🎉
