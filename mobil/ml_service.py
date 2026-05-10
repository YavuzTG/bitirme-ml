import os
import pickle
from typing import Dict, List

import numpy as np
from tensorflow.keras.models import load_model


class ModelService:
    """Shared model service for desktop UI and mobile API inference."""

    def __init__(
        self,
        model_bundle_path: str = None,
        cnn_model_path: str = None,
        lstm_model_path: str = None,
    ):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        desktop_dir = os.path.join(os.path.dirname(current_dir), "masaustu")

        self.model_bundle_path = model_bundle_path or os.path.join(desktop_dir, "trained_models.pkl")
        self.cnn_model_path = cnn_model_path or os.path.join(desktop_dir, "model_cnn.keras")
        self.lstm_model_path = lstm_model_path or os.path.join(desktop_dir, "model_lstm.keras")
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if not os.path.exists(self.model_bundle_path):
            raise FileNotFoundError(self.model_bundle_path)
        if not os.path.exists(self.cnn_model_path):
            raise FileNotFoundError(self.cnn_model_path)
        if not os.path.exists(self.lstm_model_path):
            raise FileNotFoundError(self.lstm_model_path)

        with open(self.model_bundle_path, "rb") as f:
            obj = pickle.load(f)

        self.scaler = obj["scaler"]
        self.pca = obj["pca"]
        self.svm = obj["svm"]
        self.timesteps = int(obj.get("TIMESTEPS", 5))

        classes = obj.get("classes")
        if classes is None:
            classes = np.arange(int(obj.get("num_classes", 0)))
        self.classes = [int(c) if str(c).isdigit() else str(c) for c in classes]

        self.model_cnn = load_model(self.cnn_model_path)
        self.model_lstm = load_model(self.lstm_model_path)
        self._loaded = True

    def _build_label(self, class_index: int) -> str:
        if class_index < 0 or class_index >= len(self.classes):
            return str(class_index)
        return str(self.classes[class_index])

    def predict(self, features: List[float]) -> Dict[str, Dict[str, object]]:
        self._ensure_loaded()

        x = np.array(features, dtype=float).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        x_cnn = x_scaled[..., np.newaxis]
        cnn_probs = self.model_cnn.predict(x_cnn, verbose=0)[0]
        cnn_idx = int(np.argmax(cnn_probs))

        x_pca = self.pca.transform(x_scaled)
        svm_idx = int(self.svm.predict(x_pca)[0])

        x_seq = np.tile(x_scaled, (self.timesteps, 1))[np.newaxis, ..., np.newaxis]
        lstm_probs = self.model_lstm.predict(x_seq, verbose=0)[0]
        lstm_idx = int(np.argmax(lstm_probs))

        return {
            "cnn": {
                "predicted_class_index": cnn_idx,
                "predicted_y_label": self._build_label(cnn_idx),
                "confidence": float(np.max(cnn_probs)),
                "probabilities": [float(v) for v in cnn_probs],
            },
            "svm": {
                "predicted_class_index": svm_idx,
                "predicted_y_label": str(svm_idx),
                "confidence": None,
                "probabilities": None,
            },
            "lstm": {
                "predicted_class_index": lstm_idx,
                "predicted_y_label": self._build_label(lstm_idx),
                "confidence": float(np.max(lstm_probs)),
                "probabilities": [float(v) for v in lstm_probs],
            },
        }
