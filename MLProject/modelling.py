"""
modelling.py
============
Melatih model machine learning (Random Forest) menggunakan MLflow
Tracking UI yang disimpan secara lokal, dengan autolog dari MLflow
dan TANPA hyperparameter tuning (Kriteria 2 - Basic).

Sebelum menjalankan script ini, jalankan MLflow Tracking UI terlebih dahulu:
    mlflow ui --host 127.0.0.1 --port 5000

Lalu jalankan:
    python modelling.py
"""

import os
import warnings

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "diabetes_preprocessing",
    "diabetes_preprocessing.csv",
)
TARGET_COL = "Outcome"

# Simpan seluruh artefak pada MLflow Tracking UI lokal.
# Saat dijalankan lewat `mlflow run` (CI), MLFLOW_TRACKING_URI sudah
# diatur oleh environment sehingga tidak perlu di-set ulang.
if os.environ.get("MLFLOW_RUN_ID") is None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/"))
    mlflow.set_experiment("Diabetes-Classification")

# Aktifkan autolog: parameter, metrik, dan model dicatat otomatis
mlflow.sklearn.autolog()


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="rf_baseline_autolog"):
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        print(f"Akurasi data uji: {accuracy:.4f}")


if __name__ == "__main__":
    main()
