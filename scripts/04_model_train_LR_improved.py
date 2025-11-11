import argparse
import os
import glob
import pandas as pd
import pickle
import numpy as np
import pprint
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn

import tempfile
import pickle
import os

# To run: python 04_model_train_LR.py --snapshotdate "2024-09-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize Spark
    spark = pyspark.sql.SparkSession.builder.appName("TrainLR").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- Configuration ---
    model_train_date_str = snapshotdate
    train_test_period_months = 12
    oot_period_months = 2
    train_test_ratio = 0.8

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] = oot_period_months
    config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] = config['model_train_date'] - timedelta(days=1)
    config["oot_start_date"] = config['model_train_date'] - relativedelta(months=oot_period_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_period_months)
    config["train_test_ratio"] = train_test_ratio
    pprint.pprint(config)

    # --- get label ---
    LABEL_DIR = "/app/datamart/gold/label_store"
    # Find subdirectories that look like *.parquet (folders, not files)
    subfolders = sorted([os.path.join(LABEL_DIR, d) for d in os.listdir(LABEL_DIR) if d.endswith(".parquet")])
    
    if not subfolders:
        raise FileNotFoundError(f"No label snapshot folders found in {LABEL_DIR}")
    
    print(f"📂 Found {len(subfolders)} label snapshots")
    label_store_sdf = spark.read.parquet(*subfolders)
    print("✅ row_count:", label_store_sdf.count())
        
    label_store_sdf.show(5)

    # extract label store
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))
    print("extracted labels_sdf", labels_sdf.count(), config["train_test_start_date"], config["oot_end_date"])


    # --- get features ---
    FEATURE_DIR = "/app/datamart/gold/feature_store"
    subfolders = sorted([os.path.join(FEATURE_DIR, d) for d in os.listdir(FEATURE_DIR) if d.startswith("snapshot_date=")])
    
    if not subfolders:
        raise FileNotFoundError(f"No snapshot_date folders found in {FEATURE_DIR}")
    
    print(f"📂 Found {len(subfolders)} feature snapshots")
    features_store_sdf = spark.read.parquet(*subfolders)
    print("✅ row_count:", features_store_sdf.count())

    features_store_sdf.show(5)
    
    # extract feature store
    try:
        features_sdf = features_store_sdf.filter(
            (col("snapshot_date") >= config["train_test_start_date"]) &
            (col("snapshot_date") <= config["oot_end_date"])
        )
    except Exception as e:
        print(f"⚠️ Using application_date instead of snapshot_date due to: {e}")
        features_sdf = features_store_sdf.filter(
            (col("application_date") >= config["train_test_start_date"]) &
            (col("application_date") <= config["oot_end_date"])
        )
        
    print("extracted features_sdf", features_sdf.count(), config["train_test_start_date"], config["oot_end_date"])

    # --- Prepare training data ---
    data_pdf = labels_sdf.join(features_sdf, on=["Customer_ID"], how="inner").toPandas()

    oot_pdf = data_pdf[
        (data_pdf['snapshot_date'] >= config["oot_start_date"].date()) &
        (data_pdf['snapshot_date'] <= config["oot_end_date"].date())
    ]
    train_test_pdf = data_pdf[
        (data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) &
        (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())
    ]

    feature_cols = [fe_col for fe_col in features_sdf.columns if fe_col not in ['Customer_ID', 'application_date', 'snapshot_date']]

    X_oot = oot_pdf[feature_cols]
    y_oot = oot_pdf["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols],
        train_test_pdf["label"],
        test_size=1 - config["train_test_ratio"],
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"]
    )

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}, OOT size: {X_oot.shape[0]}")

    # --- Preprocessing ---
    # --- Handle missing values ---
    X_train = X_train.fillna(0)
    X_test  = X_test.fillna(0)
    X_oot   = X_oot.fillna(0)

    # --- Scaling ---
    scaler = StandardScaler()
    transformer_stdscaler = scaler.fit(X_train)
    X_train_processed = transformer_stdscaler.transform(X_train)
    X_test_processed = transformer_stdscaler.transform(X_test)
    X_oot_processed = transformer_stdscaler.transform(X_oot)

    # --- Model training ---
    log_reg = LogisticRegression(solver='liblinear', random_state=88, max_iter=1000)
    param_dist = {
        'C': np.logspace(-3, 3, 10),
        'penalty': ['l1', 'l2']
    }

    random_search = RandomizedSearchCV(
        estimator=log_reg,
        param_distributions=param_dist,
        scoring='roc_auc',
        n_iter=10,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train_processed, y_train)
    print("Best parameters:", random_search.best_params_)
    print("Best AUC:", random_search.best_score_)

    best_model = random_search.best_estimator_

    # --- Evaluate ---
    def eval_auc(X, y):
        y_pred_proba = best_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        return auc, round(2 * auc - 1, 3)

    train_auc, train_gini = eval_auc(X_train_processed, y_train)
    test_auc, test_gini = eval_auc(X_test_processed, y_test)
    oot_auc, oot_gini = eval_auc(X_oot_processed, y_oot)

    print("Train AUC:", train_auc, "GINI:", train_gini)
    print("Test AUC:", test_auc, "GINI:", test_gini)
    print("OOT AUC:", oot_auc, "GINI:", oot_gini)

    # --- MLflow Tracking ---
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("credit_model_LR_3")
    
    with mlflow.start_run(run_name=f"run_{config['model_train_date_str']}"):
        try:
            # Log parameters
            mlflow.log_params(random_search.best_params_)
            mlflow.log_param("train_test_ratio", config["train_test_ratio"])
            mlflow.log_param("train_test_period_months", config["train_test_period_months"])
            mlflow.log_param("oot_period_months", config["oot_period_months"])
            mlflow.log_param("model_train_date_str", config["model_train_date_str"])
            
            # Log metrics
            mlflow.log_metric("train_auc", train_auc)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.log_metric("oot_auc", oot_auc)
            mlflow.log_metric("train_gini", train_gini)
            mlflow.log_metric("test_gini", test_gini)
            mlflow.log_metric("oot_gini", oot_gini)
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            print("✅ Model logged successfully")
            
            # Log scaler using temp file
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
                pickle.dump(transformer_stdscaler, f)
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "preprocessing")
            os.remove(temp_path)
            print("✅ Scaler logged successfully")
            
            # Register model
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(model_uri=model_uri, name="credit_model_LR_3")
            print(f"✅ Model registered: credit_model_LR_3 v{registered_model.version} (run: {run_id})")
            print(f"MLflow artifact URI: {mlflow.get_artifact_uri()}")
            
        except Exception as e:
            print(f"❌ MLflow logging failed: {e}")
            raise

    # --- Save model artifact ---
    model_artefact = {
        'model': best_model,
        'model_version': "credit_model_LR_" + config["model_train_date_str"].replace('-', '_'),
        'preprocessing_transformers': {'stdscaler': transformer_stdscaler},
        'data_dates': config,
        'data_stats': {
            'X_train': X_train.shape[0],
            'X_test': X_test.shape[0],
            'X_oot': X_oot.shape[0],
            'y_train': round(y_train.mean(), 2),
            'y_test': round(y_test.mean(), 2),
            'y_oot': round(y_oot.mean(), 2)
        },
        'results': {
            'auc_train': train_auc, 'auc_test': test_auc, 'auc_oot': oot_auc,
            'gini_train': train_gini, 'gini_test': test_gini, 'gini_oot': oot_gini
        },
        'hp_params': random_search.best_params_
    }

    model_bank_directory = "/opt/airflow/model_bank/"
    os.makedirs(model_bank_directory, exist_ok=True)
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"Model saved to {file_path}")

    # --- Validate reload ---
    with open(file_path, 'rb') as file:
        loaded_model_artefact = pickle.load(file)

    y_pred_proba = loaded_model_artefact['model'].predict_proba(X_oot_processed)[:, 1]
    oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
    print("Reloaded OOT AUC:", oot_auc_score)
    print("Model reloaded successfully!")

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Logistic Regression model training")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
