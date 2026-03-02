"""
modelling.py - MLflow Project Version
=======================================
Model training script untuk dijalankan via MLflow Project.
Menerima hyperparameter melalui command line arguments.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, max_error
)
import mlflow
import mlflow.sklearn
import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data():
    """Memuat dataset yang sudah dipreprocessing."""
    # Cek lokasi file (bisa di subfolder atau di root)
    if os.path.exists('auto_mpg_preprocessing/auto_mpg_train.csv'):
        train_path = 'auto_mpg_preprocessing/auto_mpg_train.csv'
        test_path = 'auto_mpg_preprocessing/auto_mpg_test.csv'
    else:
        train_path = 'auto_mpg_train.csv'
        test_path = 'auto_mpg_test.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop('mpg', axis=1)
    y_train = train_df['mpg']
    X_test = test_df.drop('mpg', axis=1)
    y_test = test_df['mpg']

    print(f"[INFO] Data dimuat: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_residual_plot(y_test, y_pred, filename="residual_plot.png"):
    """Membuat dan menyimpan residual plot."""
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', s=30)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2)
    axes[0].set_xlabel('Actual MPG')
    axes[0].set_ylabel('Predicted MPG')
    axes[0].set_title('Actual vs Predicted')

    axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename


def save_feature_importance_plot(model, feature_names, filename="feature_importance.png"):
    """Membuat dan menyimpan feature importance plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
    plt.xticks(range(len(importances)),
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename


def train_model(n_estimators, max_depth):
    """Melatih model dan log ke MLflow."""

    mlflow.set_experiment("Auto_MPG_CI")

    X_train, X_test, y_train, y_test = load_preprocessed_data()

    with mlflow.start_run():
        # --- Log Parameters ---
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)

        # --- Train Model ---
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # --- Predict ---
        y_pred = model.predict(X_test)

        # --- Log Metrics ---
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        max_err = max_error(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("max_error", max_err)

        # --- Log Model ---
        mlflow.sklearn.log_model(model, artifact_path="model")

        # --- Log Tags ---
        mlflow.set_tag("stage", "ci")
        mlflow.set_tag("dataset", "auto_mpg")

        # --- Log Artifacts ---
        residual_file = save_residual_plot(y_test, y_pred)
        mlflow.log_artifact(residual_file)

        fi_file = save_feature_importance_plot(model, X_train.columns.tolist())
        mlflow.log_artifact(fi_file)

        # Predictions CSV
        results_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_pred,
            'residual': y_test.values - y_pred
        })
        results_file = "predictions.csv"
        results_df.to_csv(results_file, index=False)
        mlflow.log_artifact(results_file)

        # Best params JSON
        params_dict = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": 42
        }
        with open("params.json", 'w') as f:
            json.dump(params_dict, f, indent=2)
        mlflow.log_artifact("params.json")

        # Print results
        print(f"\n=== Hasil Evaluasi ===")
        print(f"n_estimators : {n_estimators}")
        print(f"max_depth    : {max_depth}")
        print(f"MSE          : {mse:.4f}")
        print(f"RMSE         : {rmse:.4f}")
        print(f"MAE          : {mae:.4f}")
        print(f"R2 Score     : {r2:.4f}")
        print(f"MAPE         : {mape:.4f}")
        print(f"Max Error    : {max_err:.4f}")

        # Cleanup temp files
        for f in ['residual_plot.png', 'feature_importance.png',
                   'predictions.csv', 'params.json']:
            if os.path.exists(f):
                os.remove(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    args = parser.parse_args()

    train_model(args.n_estimators, args.max_depth)
