import sys
import os
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd
import threading
from datetime import datetime, timezone

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit,
    QFileDialog, QProgressBar, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette

# ─────────────────────────────────────────────
#  ML arka plan iş parçacığı
# ─────────────────────────────────────────────
class TrainWorker(QObject):
    log        = pyqtSignal(str)
    progress   = pyqtSignal(int)
    finished   = pyqtSignal(dict)   # {'cnn': acc, 'svm': acc, 'lstm': acc}
    error      = pyqtSignal(str)

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

    def run(self):
        try:
            # ── Gerekli kütüphaneler ──
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from sklearn.svm import SVC
            from sklearn.utils.class_weight import compute_class_weight

            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                Conv1D, MaxPooling1D, Flatten, Dense,
                Dropout, LSTM, TimeDistributed
            )
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping

            # ── Veri Yükle ──
            self.log.emit("📂 Veri okunuyor...")
            df = pd.read_csv(self.csv_path)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            self.log.emit(f"✅ Veri yüklendi — X: {X.shape}, Sınıflar: {np.unique(y)}")
            self.progress.emit(5)

            # ── Bölme & Ölçekleme ──
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)
            self.log.emit("✅ Train/Test bölündü ve ölçeklendi.")
            self.progress.emit(10)

            # Sınıf ağırlıkları
            class_weights = compute_class_weight(
                "balanced", classes=np.unique(y_train), y=y_train
            )
            cw_dict = dict(zip(np.unique(y_train), class_weights))
            num_classes = len(np.unique(y))

            # ── MODEL 1: CNN ──
            self.log.emit("\n🔵 CNN eğitimi başlıyor...")
            X_tr_cnn = X_train[..., np.newaxis]
            X_ts_cnn = X_test[..., np.newaxis]

            model_cnn = Sequential([
                Conv1D(32, 3, activation="relu", input_shape=(X_tr_cnn.shape[1], 1)),
                MaxPooling1D(2),
                Conv1D(64, 3, activation="relu"),
                MaxPooling1D(2),
                Flatten(),
                Dense(64, activation="relu"),
                Dropout(0.5),
                Dense(num_classes, activation="softmax")
            ])
            model_cnn.compile(
                optimizer=Adam(1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            es = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
            model_cnn.fit(
                X_tr_cnn, y_train,
                validation_split=0.2, epochs=50, batch_size=32,
                class_weight=cw_dict, callbacks=[es], verbose=0
            )
            cnn_acc = model_cnn.evaluate(X_ts_cnn, y_test, verbose=0)[1]
            self.log.emit(f"   CNN Accuracy: {cnn_acc:.4f}")
            self.progress.emit(45)

            # ── MODEL 2: PCA + SVM ──
            self.log.emit("\n🟡 PCA + SVM eğitimi başlıyor...")
            pca = PCA(n_components=0.95)
            X_tr_pca = pca.fit_transform(X_train)
            X_ts_pca = pca.transform(X_test)
            svm = SVC(kernel="rbf", C=10, gamma="scale")
            svm.fit(X_tr_pca, y_train)
            svm_acc = svm.score(X_ts_pca, y_test)
            self.log.emit(f"   PCA+SVM Accuracy: {svm_acc:.4f}")
            self.progress.emit(65)

            # ── MODEL 3: CNN-LSTM ──
            self.log.emit("\n🟣 CNN-LSTM eğitimi başlıyor...")
            TIMESTEPS = 5
            def make_seq(arr, t):
                return np.array([np.tile(r, (t, 1)) for r in arr])[..., np.newaxis]

            X_tr_seq = make_seq(X_train, TIMESTEPS)
            X_ts_seq = make_seq(X_test, TIMESTEPS)

            model_lstm = Sequential([
                TimeDistributed(
                    Conv1D(32, 3, activation="relu"),
                    input_shape=(TIMESTEPS, X_train.shape[1], 1)
                ),
                TimeDistributed(MaxPooling1D(2)),
                TimeDistributed(Flatten()),
                LSTM(64),
                Dense(64, activation="relu"),
                Dense(num_classes, activation="softmax")
            ])
            model_lstm.compile(
                optimizer=Adam(1e-4),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )
            model_lstm.fit(
                X_tr_seq, y_train,
                validation_split=0.2, epochs=50, batch_size=32,
                class_weight=cw_dict, callbacks=[es], verbose=0
            )
            lstm_acc = model_lstm.evaluate(X_ts_seq, y_test, verbose=0)[1]
            self.log.emit(f"   CNN-LSTM Accuracy: {lstm_acc:.4f}")
            self.progress.emit(95)

            # ── Modelleri kaydet (tahmin için) ──
            import pickle
            save = {
                "scaler": scaler, "pca": pca, "svm": svm,
                "num_classes": num_classes, "TIMESTEPS": TIMESTEPS
            }
            with open("trained_models.pkl", "wb") as f:
                pickle.dump(save, f)
            model_cnn.save("model_cnn.keras")
            model_lstm.save("model_lstm.keras")

            self.log.emit("\n✅ Modeller kaydedildi.")
            self.progress.emit(100)
            self.finished.emit({
                "cnn": cnn_acc,
                "svm": svm_acc,
                "lstm": lstm_acc
            })

        except Exception as e:
            self.error.emit(str(e))


class RemoteTrainWorker(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, owner, repo, token, workflow_file="train.yml", branch="main", data_path="BEED_Data.csv"):
        super().__init__()
        self.owner = owner
        self.repo = repo
        self.token = token
        self.workflow_file = workflow_file
        self.branch = branch
        self.data_path = data_path

    def _headers(self):
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def run(self):
        try:
            import requests

            api_base = f"https://api.github.com/repos/{self.owner}/{self.repo}"
            started_at = datetime.now(timezone.utc)

            self.log.emit("☁️ Uzak eğitim tetikleniyor (GitHub Actions)...")
            self.progress.emit(5)

            dispatch_url = f"{api_base}/actions/workflows/{self.workflow_file}/dispatches"
            payload = {
                "ref": self.branch,
                "inputs": {"data_path": self.data_path},
            }
            dispatch_resp = requests.post(dispatch_url, headers=self._headers(), json=payload, timeout=30)
            if dispatch_resp.status_code != 204:
                raise RuntimeError(f"Workflow tetiklenemedi: {dispatch_resp.status_code} {dispatch_resp.text}")

            self.log.emit("✅ Workflow tetiklendi. Run bilgisi bekleniyor...")
            self.progress.emit(15)

            run_id = None
            for _ in range(30):
                runs_url = f"{api_base}/actions/workflows/{self.workflow_file}/runs"
                runs_resp = requests.get(
                    runs_url,
                    headers=self._headers(),
                    params={"event": "workflow_dispatch", "branch": self.branch, "per_page": 20},
                    timeout=30,
                )
                if runs_resp.status_code != 200:
                    raise RuntimeError(f"Run listesi alınamadı: {runs_resp.status_code} {runs_resp.text}")

                runs = runs_resp.json().get("workflow_runs", [])
                for run in runs:
                    created_at = run.get("created_at")
                    if not created_at:
                        continue
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if created_dt >= started_at:
                        run_id = run.get("id")
                        break

                if run_id:
                    break

                self.log.emit("⏳ Run henüz görünmedi, tekrar kontrol ediliyor...")
                time.sleep(5)

            if not run_id:
                raise RuntimeError("Workflow run bulunamadı. GitHub Actions sekmesinden kontrol et.")

            self.log.emit(f"🚀 Run başladı (ID: {run_id}). Eğitim sürüyor...")
            self.progress.emit(25)

            progress_value = 25
            run_url = f"{api_base}/actions/runs/{run_id}"
            while True:
                run_resp = requests.get(run_url, headers=self._headers(), timeout=30)
                if run_resp.status_code != 200:
                    raise RuntimeError(f"Run durumu alınamadı: {run_resp.status_code} {run_resp.text}")

                run_info = run_resp.json()
                status = run_info.get("status")
                conclusion = run_info.get("conclusion")

                if status == "completed":
                    if conclusion != "success":
                        html_url = run_info.get("html_url", "")
                        raise RuntimeError(f"Eğitim başarısız: {conclusion}. Detay: {html_url}")
                    break

                progress_value = min(progress_value + 5, 80)
                self.progress.emit(progress_value)
                self.log.emit(f"⏳ Workflow durumu: {status}")
                time.sleep(10)

            self.log.emit("📦 Artifact indiriliyor...")
            self.progress.emit(85)

            artifacts_url = f"{api_base}/actions/runs/{run_id}/artifacts"
            artifacts_resp = requests.get(artifacts_url, headers=self._headers(), timeout=30)
            if artifacts_resp.status_code != 200:
                raise RuntimeError(f"Artifact listesi alınamadı: {artifacts_resp.status_code} {artifacts_resp.text}")

            artifacts = artifacts_resp.json().get("artifacts", [])
            artifact = next((a for a in artifacts if a.get("name") == "beed-model-bundle"), None)
            if artifact is None:
                if not artifacts:
                    raise RuntimeError("Artifact bulunamadı.")
                artifact = artifacts[0]

            dl_url = artifact.get("archive_download_url")
            if not dl_url:
                raise RuntimeError("Artifact indirme bağlantısı alınamadı.")

            dl_resp = requests.get(dl_url, headers=self._headers(), timeout=60, allow_redirects=True)
            if dl_resp.status_code != 200:
                raise RuntimeError(f"Artifact indirilemedi: {dl_resp.status_code} {dl_resp.text}")

            wanted_files = {"model_cnn.keras", "model_lstm.keras", "trained_models.pkl", "metrics.json"}
            with zipfile.ZipFile(io.BytesIO(dl_resp.content)) as zf:
                for member in zf.namelist():
                    name = os.path.basename(member)
                    if name in wanted_files:
                        with zf.open(member) as src, open(name, "wb") as dst:
                            dst.write(src.read())

            metrics = {"cnn": 0.0, "svm": 0.0, "lstm": 0.0}
            if os.path.exists("metrics.json"):
                with open("metrics.json", "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                metrics["cnn"] = float(loaded.get("cnn", 0.0))
                metrics["svm"] = float(loaded.get("svm", 0.0))
                metrics["lstm"] = float(loaded.get("lstm", 0.0))

            self.log.emit("✅ Uzak eğitim tamamlandı ve modeller indirildi.")
            self.progress.emit(100)
            self.finished.emit(metrics)

        except ImportError:
            self.error.emit("'requests' paketi eksik. Kur: pip install requests")
        except Exception as e:
            self.error.emit(str(e))


class PredictWorker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, features: list):
        super().__init__()
        self.features = features

    def run(self):
        try:
            import pickle
            from tensorflow.keras.models import load_model

            with open("trained_models.pkl", "rb") as f:
                obj = pickle.load(f)
            scaler     = obj["scaler"]
            pca        = obj["pca"]
            svm        = obj["svm"]
            TIMESTEPS  = obj["TIMESTEPS"]

            x = np.array(self.features).reshape(1, -1)
            x_scaled = scaler.transform(x)

            # CNN
            model_cnn  = load_model("model_cnn.keras")
            x_cnn      = x_scaled[..., np.newaxis]
            pred_cnn   = int(np.argmax(model_cnn.predict(x_cnn, verbose=0)))

            # SVM
            x_pca      = pca.transform(x_scaled)
            pred_svm   = int(svm.predict(x_pca)[0])

            # CNN-LSTM
            model_lstm = load_model("model_lstm.keras")
            x_seq      = np.tile(x_scaled, (TIMESTEPS, 1))[np.newaxis, ..., np.newaxis]
            pred_lstm  = int(np.argmax(model_lstm.predict(x_seq, verbose=0)))

            self.finished.emit({
                "cnn": pred_cnn,
                "svm": pred_svm,
                "lstm": pred_lstm
            })
        except FileNotFoundError:
            self.error.emit("Eğitilmiş model bulunamadı!\nLütfen önce 'Model Eğitimi' sekmesinden bir model eğitin.")
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────
#  Ana Pencere
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BEED — Sinyal Sınıflandırma")
        self.setMinimumSize(820, 620)
        self._apply_style()

        tabs = QTabWidget()
        tabs.addTab(self._build_train_tab(), "🔧  Model Eğitimi")
        tabs.addTab(self._build_predict_tab(), "🔍  Tahmin")
        self.setCentralWidget(tabs)

    # ── Stil ──────────────────────────────────
    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1e1e2e; color: #cdd6f4; }
            QTabWidget::pane { border: 1px solid #313244; background: #1e1e2e; }
            QTabBar::tab {
                background: #313244; color: #cdd6f4;
                padding: 8px 20px; border-radius: 4px 4px 0 0;
            }
            QTabBar::tab:selected { background: #89b4fa; color: #1e1e2e; font-weight: bold; }
            QPushButton {
                background: #89b4fa; color: #1e1e2e;
                border: none; padding: 8px 18px;
                border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background: #b4befe; }
            QPushButton:disabled { background: #45475a; color: #6c7086; }
            QLineEdit {
                background: #313244; border: 1px solid #45475a;
                border-radius: 4px; padding: 5px 8px; color: #cdd6f4;
            }
            QLineEdit:focus { border: 1px solid #89b4fa; }
            QTextEdit {
                background: #181825; border: 1px solid #313244;
                border-radius: 6px; font-family: Consolas, monospace; font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #45475a; border-radius: 8px;
                margin-top: 10px; padding: 10px;
                font-weight: bold; color: #89b4fa;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QProgressBar {
                background: #313244; border-radius: 6px; height: 18px;
                text-align: center; color: #1e1e2e;
            }
            QProgressBar::chunk { background: #a6e3a1; border-radius: 6px; }
            QTableWidget {
                background: #181825; gridline-color: #313244;
                border: 1px solid #313244; border-radius: 6px;
            }
            QHeaderView::section { background: #313244; color: #89b4fa; padding: 5px; }
            QLabel#result_label {
                font-size: 18px; font-weight: bold;
                background: #313244; border-radius: 8px; padding: 10px;
            }
        """)

    # ── Tab 1: Eğitim ─────────────────────────
    def _build_train_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Dosya seçimi
        file_group = QGroupBox("Veri Dosyası")
        fl = QHBoxLayout(file_group)
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("CSV dosyasını seçin...")
        self.file_path_edit.setReadOnly(True)
        browse_btn = QPushButton("📁 Gözat")
        browse_btn.clicked.connect(self._browse_csv)
        fl.addWidget(self.file_path_edit)
        fl.addWidget(browse_btn)
        layout.addWidget(file_group)

        # Eğit butonu + progress
        ctrl_group = QGroupBox("Eğitim")
        cl = QVBoxLayout(ctrl_group)
        self.train_btn = QPushButton("▶  Modelleri Eğit")
        self.train_btn.setFixedHeight(42)
        self.train_btn.clicked.connect(self._start_training)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        cl.addWidget(self.train_btn)
        cl.addWidget(self.progress_bar)
        layout.addWidget(ctrl_group)

        # Sonuç tablosu
        res_group = QGroupBox("Doğruluk Sonuçları")
        rl = QVBoxLayout(res_group)
        self.acc_table = QTableWidget(3, 2)
        self.acc_table.setHorizontalHeaderLabels(["Model", "Accuracy"])
        self.acc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.acc_table.setFixedHeight(110)
        self.acc_table.verticalHeader().setVisible(False)
        for i, name in enumerate(["CNN", "PCA + SVM", "CNN-LSTM"]):
            self.acc_table.setItem(i, 0, QTableWidgetItem(name))
            self.acc_table.setItem(i, 1, QTableWidgetItem("—"))
        rl.addWidget(self.acc_table)
        layout.addWidget(res_group)

        # Log alanı
        log_group = QGroupBox("Eğitim Logu")
        ll = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        ll.addWidget(self.log_edit)
        layout.addWidget(log_group, stretch=1)

        return w

    # ── Tab 2: Tahmin ─────────────────────────
    def _build_predict_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        inp_group = QGroupBox("Kişi Verisini Girin  (X1 – X16)")
        grid = QGridLayout(inp_group)
        self.feat_inputs = []
        for i in range(16):
            lbl = QLabel(f"X{i+1}:")
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            edit = QLineEdit()
            edit.setPlaceholderText("0")
            grid.addWidget(lbl,  i // 4, (i % 4) * 2)
            grid.addWidget(edit, i // 4, (i % 4) * 2 + 1)
            self.feat_inputs.append(edit)
        layout.addWidget(inp_group)

        self.predict_btn = QPushButton("🔍  Tahmin Et")
        self.predict_btn.setFixedHeight(42)
        self.predict_btn.clicked.connect(self._start_predict)
        layout.addWidget(self.predict_btn)

        # Sonuç kutuları
        res_group = QGroupBox("Tahmin Sonuçları")
        rl = QHBoxLayout(res_group)
        self.res_cnn  = self._make_result_card("CNN",       "#89b4fa")
        self.res_svm  = self._make_result_card("PCA+SVM",   "#f9e2af")
        self.res_lstm = self._make_result_card("CNN-LSTM",  "#cba6f7")
        rl.addWidget(self.res_cnn[0])
        rl.addWidget(self.res_svm[0])
        rl.addWidget(self.res_lstm[0])
        layout.addWidget(res_group)

        layout.addStretch()
        return w

    def _make_result_card(self, title, color):
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{ background: #313244; border-radius: 10px; padding: 8px; }}
        """)
        vl = QVBoxLayout(frame)
        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignCenter)
        title_lbl.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")
        val_lbl = QLabel("—")
        val_lbl.setObjectName("result_label")
        val_lbl.setAlignment(Qt.AlignCenter)
        val_lbl.setStyleSheet(f"""
            font-size: 36px; font-weight: bold;
            color: {color}; background: #1e1e2e;
            border-radius: 8px; padding: 12px;
        """)
        vl.addWidget(title_lbl)
        vl.addWidget(val_lbl)
        return frame, val_lbl

    # ── Olaylar ───────────────────────────────
    def _browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "CSV Seç", "", "CSV Dosyaları (*.csv)"
        )
        if path:
            self.file_path_edit.setText(path)

    def _start_training(self):
        gh_owner = os.getenv("GH_OWNER", "").strip()
        gh_repo = os.getenv("GH_REPO", "").strip()
        gh_token = os.getenv("GH_TOKEN", "").strip()

        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_edit.clear()

        self._train_thread = QThread()

        if gh_owner and gh_repo and gh_token:
            data_path = self.file_path_edit.text().strip() or "BEED_Data.csv"
            data_path = os.path.basename(data_path)
            workflow_file = os.getenv("GH_WORKFLOW_FILE", "train.yml").strip() or "train.yml"
            branch = os.getenv("GH_BRANCH", "main").strip() or "main"

            self.log_edit.append("☁️ Mod: Uzak eğitim (GitHub Actions)")
            self.log_edit.append(f"📄 Veri yolu (repo içi): {data_path}")

            self._train_worker = RemoteTrainWorker(
                owner=gh_owner,
                repo=gh_repo,
                token=gh_token,
                workflow_file=workflow_file,
                branch=branch,
                data_path=data_path,
            )
        else:
            path = self.file_path_edit.text().strip()
            if not path or not os.path.exists(path):
                self.train_btn.setEnabled(True)
                QMessageBox.warning(self, "Hata", "Lütfen geçerli bir CSV dosyası seçin.")
                return

            self.log_edit.append("💻 Mod: Yerel eğitim")
            self._train_worker = TrainWorker(path)

        self._train_worker.moveToThread(self._train_thread)

        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.log.connect(self.log_edit.append)
        self._train_worker.progress.connect(self.progress_bar.setValue)
        self._train_worker.finished.connect(self._on_train_done)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.finished.connect(self._train_thread.quit)
        self._train_worker.error.connect(self._train_thread.quit)
        self._train_thread.finished.connect(lambda: self.train_btn.setEnabled(True))

        self._train_thread.start()

    def _on_train_done(self, accs):
        for i, (name, key) in enumerate([("CNN","cnn"),("PCA + SVM","svm"),("CNN-LSTM","lstm")]):
            item = QTableWidgetItem(f"{accs[key]*100:.2f} %")
            item.setTextAlignment(Qt.AlignCenter)
            item.setForeground(QColor("#a6e3a1"))
            self.acc_table.setItem(i, 1, item)
        self.log_edit.append("\n🎉 Eğitim tamamlandı!")

    def _on_train_error(self, msg):
        self.log_edit.append(f"\n❌ HATA: {msg}")
        QMessageBox.critical(self, "Eğitim Hatası", msg)

    def _start_predict(self):
        vals = []
        for i, edit in enumerate(self.feat_inputs):
            txt = edit.text().strip()
            if txt == "":
                QMessageBox.warning(self, "Eksik Veri", f"X{i+1} boş bırakılamaz.")
                return
            try:
                vals.append(float(txt))
            except ValueError:
                QMessageBox.warning(self, "Geçersiz Değer", f"X{i+1} sayısal olmalıdır.")
                return

        self.predict_btn.setEnabled(False)
        for _, lbl in [self.res_cnn, self.res_svm, self.res_lstm]:
            lbl.setText("⏳")

        self._pred_thread = QThread()
        self._pred_worker = PredictWorker(vals)
        self._pred_worker.moveToThread(self._pred_thread)

        self._pred_thread.started.connect(self._pred_worker.run)
        self._pred_worker.finished.connect(self._on_predict_done)
        self._pred_worker.error.connect(self._on_predict_error)
        self._pred_worker.finished.connect(self._pred_thread.quit)
        self._pred_worker.error.connect(self._pred_thread.quit)
        self._pred_thread.finished.connect(lambda: self.predict_btn.setEnabled(True))

        self._pred_thread.start()

    def _on_predict_done(self, preds):
        self.res_cnn[1].setText(str(preds["cnn"]))
        self.res_svm[1].setText(str(preds["svm"]))
        self.res_lstm[1].setText(str(preds["lstm"]))

    def _on_predict_error(self, msg):
        for _, lbl in [self.res_cnn, self.res_svm, self.res_lstm]:
            lbl.setText("❌")
        QMessageBox.critical(self, "Tahmin Hatası", msg)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
