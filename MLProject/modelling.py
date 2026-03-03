"""
modelling.py
Model Machine Learning - Auto MPG Regression (Basic)
=====================================================
Melatih model Random Forest Regressor menggunakan MLflow autolog.
Artefak disimpan di MLflow Tracking UI secara lokal (localhost).

Cara menjalankan:
1. Buka terminal, jalankan: mlflow ui
2. Buka terminal lain, jalankan: python modelling.py
3. Buka browser: http://127.0.0.1:5000
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data():
    """Memuat dataset yang sudah dipreprocessing."""
    train_df = pd.read_csv('auto_mpg_preprocessing/auto_mpg_train.csv')
    test_df = pd.read_csv('auto_mpg_preprocessing/auto_mpg_test.csv')
    
    X_train = train_df.drop('mpg', axis=1)
    y_train = train_df['mpg']
    X_test = test_df.drop('mpg', axis=1)
    y_test = test_df['mpg']
    
    print(f"[INFO] Data dimuat: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_model():
    """Melatih model dengan MLflow autolog."""
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Auto_MPG_Regression")
    
    # Aktifkan autolog (WAJIB untuk kriteria Basic)
    mlflow.sklearn.autolog()
    
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    with mlflow.start_run(run_name="RandomForest_AutoLog"):
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Hitung metrik (autolog sudah log otomatis, ini untuk print saja)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n=== Hasil Evaluasi ===")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"R2   : {r2:.4f}")
        
        print(f"\n[INFO] Model dan artefak tersimpan di MLflow (autolog).")
        print(f"[INFO] Buka http://127.0.0.1:5000 untuk melihat dashboard.")


if __name__ == '__main__':
    train_model()
    
