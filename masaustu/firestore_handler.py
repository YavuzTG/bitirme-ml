from datetime import datetime

from firebase_config import firestore_db


class FirestoreHandler:
    """Cloud Firestore işlemleri"""

    @staticmethod
    def _ready():
        return firestore_db is not None

    @staticmethod
    def save_model_metadata(user_email, model_name, accuracies):
        try:
            if not FirestoreHandler._ready():
                return None

            payload = {
                "name": model_name,
                "trainedBy": user_email,
                "accuracy": accuracies,
                "createdAt": datetime.utcnow().isoformat(),
                "status": "ready",
            }
            ref = firestore_db.collection("models").add(payload)
            return ref[1].id
        except Exception as e:
            print(f"[WARN] Model kaydetme hatasi: {e}")
            return None

    @staticmethod
    def get_all_models():
        try:
            if not FirestoreHandler._ready():
                return {}

            out = {}
            for doc in firestore_db.collection("models").stream():
                out[doc.id] = doc.to_dict()
            return out
        except Exception as e:
            print(f"[WARN] Modelleri getirme hatasi: {e}")
            return {}

    @staticmethod
    def get_user_models(user_email):
        try:
            if not FirestoreHandler._ready():
                return {}

            out = {}
            query = firestore_db.collection("models").where("trainedBy", "==", user_email).stream()
            for doc in query:
                out[doc.id] = doc.to_dict()
            return out
        except Exception as e:
            print(f"[WARN] Kullanici modelleri hatasi: {e}")
            return {}

    @staticmethod
    def save_prediction(user_email, model_id, input_features, predictions):
        try:
            if not FirestoreHandler._ready():
                return None

            payload = {
                "userId": user_email,
                "modelId": model_id,
                "input": input_features,
                "result": predictions,
                "timestamp": datetime.utcnow().isoformat(),
            }
            ref = firestore_db.collection("predictions").add(payload)
            return ref[1].id
        except Exception as e:
            print(f"[WARN] Tahmin kaydetme hatasi: {e}")
            return None

    @staticmethod
    def get_user_predictions(user_email, limit=10):
        try:
            if not FirestoreHandler._ready():
                return {}

            out = {}
            query = (
                firestore_db.collection("predictions")
                .where("userId", "==", user_email)
                .order_by("timestamp")
                .limit(limit)
                .stream()
            )
            for doc in query:
                out[doc.id] = doc.to_dict()
            return out
        except Exception as e:
            print(f"[WARN] Tahmin getirme hatasi: {e}")
            return {}

    @staticmethod
    def get_prediction_stats(user_email):
        try:
            preds = FirestoreHandler.get_user_predictions(user_email, limit=1000)
            today = datetime.utcnow().date().isoformat()

            total = 0
            today_count = 0
            for pred in preds.values():
                total += 1
                if str(pred.get("timestamp", ""))[:10] == today:
                    today_count += 1

            return {"total": total, "today": today_count}
        except Exception as e:
            print(f"[WARN] Istatistik hatasi: {e}")
            return {"total": 0, "today": 0}

    @staticmethod
    def delete_model(model_id):
        try:
            if not FirestoreHandler._ready():
                return False

            firestore_db.collection("models").document(model_id).delete()
            return True
        except Exception as e:
            print(f"[WARN] Model silme hatasi: {e}")
            return False

    @staticmethod
    def save_user_info(user_email, is_admin=False):
        try:
            if not FirestoreHandler._ready():
                return False

            user_id = user_email.replace("@", "_").replace(".", "_")
            payload = {
                "email": user_email,
                "isAdmin": is_admin,
                "lastLogin": datetime.utcnow().isoformat(),
            }
            firestore_db.collection("users").document(user_id).set(payload, merge=True)
            return True
        except Exception as e:
            print(f"[WARN] Kullanici kaydetme hatasi: {e}")
            return False
