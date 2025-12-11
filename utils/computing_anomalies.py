"""
Docstring for utils.computing_anomalies
1. calculate matrices sensitive to synthesis flow 
2. phase spectrun deviation using hailbert transform
3. entropy on embeddings from pre-trained model
methods: isolation forest, auto encoder trained in real audio 
"""

import numpy as np
from scipy.signal import hilbert
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import torch
from transformers import AutoModel, AutoTokenizer


class AnomalyDetector:
    def __init__(self, model_name="bert-base-uncased"):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Tools for feature scaling and anomaly detection
        self.scaler = StandardScaler()
        self.forest = IsolationForest(contamination=0.1)

    def get_phase_deviation(self, audio_signal):
        # Use Hilbert transform to get the analytic signal
        analytic = hilbert(audio_signal)

        # Find phase information
        phase = np.angle(analytic)

        # Unwrap so values don't jump from +pi to -pi
        unwrapped_phase = np.unwrap(phase)

        # Standard deviation tells us how "weird" the phase is
        return np.std(unwrapped_phase)

    def get_entropy(self, vector):
        # Convert to probabilities
        exp_values = np.exp(vector)
        probs = exp_values / np.sum(exp_values)

        # Entropy formula
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy

    def fit(self, real_audio_list):
        all_features = []

        for audio in real_audio_list:
            # Phase deviation from Hilbert transform
            phase_dev = self.get_phase_deviation(audio)

            # Convert audio signal into text tokens (BERT expects text)
            tokens = self.tokenizer(audio, return_tensors="pt", padding=True, truncation=True)

            # Get embeddings from the model
            with torch.no_grad():
                output = self.model(**tokens)

            # Average all token embeddings to make a single vector
            embed = output.last_hidden_state.mean(dim=1).numpy()[0]

            # Entropy of this vector
            entropy_value = self.get_entropy(embed)

            # Store both features
            all_features.append([phase_dev, entropy_value])

        # Normalize features
        all_features = np.array(all_features)
        scaled = self.scaler.fit_transform(all_features)

        # Train the Isolation Forest model
        self.forest.fit(scaled)

    def predict(self, audio_signal):
        # Compute features for one audio signal
        phase_dev = self.get_phase_deviation(audio_signal)

        tokens = self.tokenizer(audio_signal, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            output = self.model(**tokens)

        embed = output.last_hidden_state.mean(dim=1).numpy()[0]
        entropy_value = self.get_entropy(embed)

        # Prepare feature for the model
        feature = np.array([[phase_dev, entropy_value]])
        feature = self.scaler.transform(feature)

        # Higher score = more normal, lower score = more anomaly
        score = self.forest.decision_function(feature)

        return score
