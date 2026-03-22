import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, LSTM, MaxPooling1D, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_and_train(data_path: str):
    start_time = time.time()

    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    num_classes = len(np.unique(y))

    # CNN
    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn = X_test[..., np.newaxis]

    model_cnn = Sequential([
        Conv1D(32, 3, activation="relu", input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model_cnn.compile(
        optimizer=Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

    model_cnn.fit(
        X_train_cnn,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0,
    )

    cnn_acc = float(model_cnn.evaluate(X_test_cnn, y_test, verbose=0)[1])

    # PCA + SVM
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm = SVC(kernel="rbf", C=10, gamma="scale")
    svm.fit(X_train_pca, y_train)
    svm_acc = float(svm.score(X_test_pca, y_test))

    # CNN-LSTM
    timesteps = 5

    def make_sequence(arr, t):
        return np.array([np.tile(row, (t, 1)) for row in arr])

    X_train_seq = make_sequence(X_train, timesteps)[..., np.newaxis]
    X_test_seq = make_sequence(X_test, timesteps)[..., np.newaxis]

    model_cnn_lstm = Sequential([
        TimeDistributed(
            Conv1D(32, 3, activation="relu"),
            input_shape=(timesteps, X_train.shape[1], 1),
        ),
        TimeDistributed(MaxPooling1D(2)),
        TimeDistributed(Flatten()),
        LSTM(64),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    model_cnn_lstm.compile(
        optimizer=Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model_cnn_lstm.fit(
        X_train_seq,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=0,
    )

    lstm_acc = float(model_cnn_lstm.evaluate(X_test_seq, y_test, verbose=0)[1])

    model_cnn.save("model_cnn.keras")
    model_cnn_lstm.save("model_lstm.keras")

    with open("trained_models.pkl", "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "pca": pca,
                "svm": svm,
                "num_classes": num_classes,
                "TIMESTEPS": timesteps,
            },
            f,
        )

    metrics = {
        "cnn": cnn_acc,
        "svm": svm_acc,
        "lstm": lstm_acc,
        "duration_seconds": round(time.time() - start_time, 2),
        "data_path": data_path,
    }
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Training completed")
    print(json.dumps(metrics, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="BEED_Data.csv", help="Path to dataset CSV")
    args = parser.parse_args()
    build_and_train(args.data)


if __name__ == "__main__":
    main()
