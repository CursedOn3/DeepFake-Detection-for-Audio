# Deepfake Audio Detection - Setup Guide

## 📋 What You Need to Do

### 1. ✅ Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

### 2. 📁 Prepare Your Dataset

Create this folder structure in the `data/` directory:

```
data/
├── train/
│   ├── real/    # Put REAL audio files here (.wav, .mp3, .flac)
│   │   ├── real_audio_1.wav
│   │   ├── real_audio_2.wav
│   │   └── ...
│   └── fake/    # Put FAKE/DEEPFAKE audio files here
│       ├── fake_audio_1.wav
│       ├── fake_audio_2.wav
│       └── ...
└── val/
    ├── real/    # Validation real audio
    │   └── ...
    └── fake/    # Validation fake audio
        └── ...
```

**Where to get datasets:**
- [ASVspoof](https://www.asvspoof.org/) - Audio spoofing dataset
- [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) - Fake audio-visual dataset
- [In-the-Wild](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) - Real-world deepfakes
- [WaveFake](https://github.com/RUB-SysSec/WaveFake) - Synthetic speech dataset

### 3. 🎯 Train the Model

```bash
# Basic training
python train.py --data_dir data/ --batch_size 16 --num_epochs 50

# Advanced training with custom settings
python train.py \
    --data_dir data/ \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.0001 \
    --embedding_size 256 \
    --num_transformer_layers 6

# Resume from checkpoint
python train.py --resume outputs/checkpoint_best.pth
```

**Training tips:**
- Start with small batch size (8-16) if you have limited GPU memory
- Training takes time! Expect 2-4 hours for 50 epochs on a good GPU
- Monitor progress with TensorBoard: `tensorboard --logdir outputs/logs/`

### 4. 🔍 Test Detection

```bash
# After training, test on new audio files
python main.py --audio test_audio.wav --model outputs/checkpoint_best.pth

# Test on multiple files
python main.py --audio test_folder/ --batch --model outputs/checkpoint_best.pth

# Adjust detection threshold (default 0.5)
python main.py --audio test.wav --threshold 0.6 --model outputs/checkpoint_best.pth
```

### 5. 📊 Evaluate Performance

Create a test script to evaluate on a test set:

```python
from main import SimpleDeepfakeDetector
from data_loader import DeepfakeAudioDataset
from torch.utils.data import DataLoader

# Load test data
test_dataset = DeepfakeAudioDataset('data/', split='test', augment=False)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load trained model
detector = SimpleDeepfakeDetector(model_weights_path='outputs/checkpoint_best.pth')

# Evaluate
correct = 0
total = 0

for batch in test_loader:
    # Get predictions
    # ... (add your evaluation code here)
    pass
```

## 🚀 Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify installation: `python verify_installation.py`
- [ ] Download or collect real and fake audio samples
- [ ] Organize audio files in `data/train/` and `data/val/` folders
- [ ] Train the model: `python train.py`
- [ ] Monitor training: `tensorboard --logdir outputs/logs/`
- [ ] Test detection: `python main.py --audio test.wav --model outputs/checkpoint_best.pth`

## ⚠️ Important Notes

### Model Performance
- **The model needs to be TRAINED before it works!**
- With random weights, predictions are meaningless
- You need at least 500-1000 real and fake audio samples for decent training
- More data = better performance

### Training Requirements
- **GPU**: Highly recommended (10-20x faster than CPU)
- **RAM**: At least 8GB
- **Storage**: Depends on dataset size (expect 5-50GB)
- **Time**: 2-4 hours on GPU, 20-40 hours on CPU

### Dataset Quality Matters
- Use high-quality audio (not compressed too much)
- Balance real and fake samples (50/50 ratio)
- Include diverse speakers and scenarios
- Ensure fake audio is from various synthesis methods (TTS, voice conversion, etc.)

## 🐛 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### "CUDA out of memory"
Reduce batch size:
```bash
python train.py --batch_size 8
```

### "No data found"
Make sure your folder structure matches the expected format:
```
data/train/real/  ← audio files here
data/train/fake/  ← audio files here
```

### Training loss not decreasing
- Check if data is loaded correctly
- Verify labels are correct (real=0, fake=1)
- Try lower learning rate: `--learning_rate 0.00001`
- Ensure sufficient training data (>500 samples per class)

## 📚 Next Steps

Once your model is trained and working:

1. **Collect more data** to improve accuracy
2. **Experiment with hyperparameters** (learning rate, model size, etc.)
3. **Test on different deepfake types** (TTS, voice cloning, etc.)
4. **Deploy the model** as an API or web service
5. **Fine-tune on specific datasets** for your use case

## 🆘 Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify all dependencies are installed
3. Ensure your dataset is correctly structured
4. Check GitHub issues for similar problems
5. Open a new issue with details about your problem

---

**Good luck! 🎉**
