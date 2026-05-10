"""
Convert Keras CNN model to a single TFLite with preprocessing.
Usage:
    python masaustu/convert_to_tflite.py --output ../mobile_app/assets/model_cnn.tflite

This script expects the following files under `masaustu/`:
 - trained_models.pkl (contains scaler and pca etc.)
 - model_cnn.keras

It will build a small wrapper Keras model that applies the scaler
and reshaping expected by the CNN, save a combined Keras model and
convert to TFLite. Optionally quantize (commented) — see notes below.
"""
import argparse
import os
import pickle
import numpy as np
import tensorflow as tf


def build_model_with_preproc(desktop_dir, cnn_path, bundle_path):
    # load sklearn objects
    with open(bundle_path, "rb") as f:
        obj = pickle.load(f)

    scaler = obj["scaler"]
    if hasattr(scaler, "mean_"):
        mean_val = scaler.mean_
    elif hasattr(scaler, "mean"):
        mean_val = scaler.mean
    else:
        raise AttributeError("Scaler object has neither 'mean_' nor 'mean' attribute")
    if hasattr(scaler, "scale_"):
        scale_val = scaler.scale_
    elif hasattr(scaler, "scale"):
        scale_val = scaler.scale
    else:
        raise AttributeError("Scaler object has neither 'scale_' nor 'scale' attribute")

    mean = np.asarray(mean_val).astype(np.float32)
    scale = np.asarray(scale_val).astype(np.float32)

    # load base cnn model
    base = tf.keras.models.load_model(cnn_path)

    # Input: raw 16-feature vector
    inp = tf.keras.Input(shape=(16,), dtype=tf.float32, name="raw_input")

    # scaler: (x - mean) / scale
    def apply_scaler(x):
        return (x - mean) / scale

    x = tf.keras.layers.Lambda(lambda z: apply_scaler(z), name="scaler")(inp)

    # reshape if base expects (batch,16,1)
    x = tf.keras.layers.Reshape((16, 1), name="reshape_for_cnn")(x)

    out = base(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="cnn_with_preproc")
    return model


def build_lstm_with_preproc(desktop_dir, lstm_path, bundle_path):
    # load sklearn objects
    with open(bundle_path, "rb") as f:
        obj = pickle.load(f)

    scaler = obj["scaler"]
    if hasattr(scaler, "mean_"):
        mean_val = scaler.mean_
    elif hasattr(scaler, "mean"):
        mean_val = scaler.mean
    else:
        raise AttributeError("Scaler object has neither 'mean_' nor 'mean' attribute")
    if hasattr(scaler, "scale_"):
        scale_val = scaler.scale_
    elif hasattr(scaler, "scale"):
        scale_val = scaler.scale
    else:
        raise AttributeError("Scaler object has neither 'scale_' nor 'scale' attribute")

    mean = np.asarray(mean_val).astype(np.float32)
    scale = np.asarray(scale_val).astype(np.float32)

    timesteps = int(obj.get("TIMESTEPS", 5))

    base = tf.keras.models.load_model(lstm_path)

    inp = tf.keras.Input(shape=(16,), dtype=tf.float32, name="raw_input")

    def apply_scaler(x):
        return (x - mean) / scale

    x = tf.keras.layers.Lambda(lambda z: apply_scaler(z), name="scaler")(inp)
    # expand to (timesteps, 16) then add batch and channel dims
    def tile_timesteps(x):
        # x shape: (batch, 16)
        x = tf.repeat(tf.expand_dims(x, 1), repeats=timesteps, axis=1)
        # result shape: (batch, timesteps, 16)
        x = tf.expand_dims(x, -1)  # (batch, timesteps, 16, 1)
        return x

    x = tf.keras.layers.Lambda(lambda z: tile_timesteps(z), name="tile_timesteps")(x)
    out = base(x)
    model = tf.keras.Model(inputs=inp, outputs=out, name="lstm_with_preproc")
    return model


def convert_to_tflite(keras_model, out_path, quantize=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # representative dataset required for full integer quantization
        def rep():
            for _ in range(100):
                yield [np.random.rand(16).astype(np.float32)]
        converter.representative_dataset = rep
        # Keep default ops; you can force INT8 if desired
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Wrote TFLite model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desktop_dir", default=os.path.dirname(__file__), help="Desktop dir (masaustu)")
    parser.add_argument("--cnn", default=os.path.join(os.path.dirname(__file__), "model_cnn.keras"), help="Path to model_cnn.keras")
    parser.add_argument("--bundle", default=os.path.join(os.path.dirname(__file__), "trained_models.pkl"), help="Path to trained_models.pkl")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "../mobile_app/assets/model_cnn.tflite"), help="Output tflite path")
    parser.add_argument("--quantize", action="store_true", help="Enable default quantization (requires representative dataset)")
    args = parser.parse_args()

    desktop_dir = args.desktop_dir
    cnn_path = args.cnn
    bundle_path = args.bundle
    out_path = args.output

    import traceback

    try:
        print("Building Keras models with preprocessing...")
        # CNN
        cnn_model = build_model_with_preproc(desktop_dir, cnn_path, bundle_path)
        # LSTM - try to find lstm model next to cnn
        lstm_path = os.path.join(desktop_dir, "model_lstm.keras")
        lstm_model = None
        if os.path.exists(lstm_path):
            lstm_model = build_lstm_with_preproc(desktop_dir, lstm_path, bundle_path)

        print("Converting to TFLite (quantize=%s)..." % args.quantize)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # write cnn tflite
        cnn_out = out_path
        convert_to_tflite(cnn_model, cnn_out, quantize=args.quantize)

        # write lstm tflite if available
        if lstm_model is not None:
            lstm_out = os.path.join(os.path.dirname(out_path), "model_lstm.tflite")
            convert_to_tflite(lstm_model, lstm_out, quantize=args.quantize)

        print("Done. Test the TFLite models locally before publishing the app.")
    except Exception as exc:
        print("ERROR during conversion:")
        traceback.print_exc()
        raise SystemExit(1)
