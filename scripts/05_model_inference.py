import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("BuildInference") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# --- set up config ---
config = {}
config["snapshot_date_str"] = "2024-09-01"
config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
config["model_name"] = "LR"

#---------------------------------------------------
# LOAD MODEL
#---------------------------------------------------

def load_model(model_name="credit_model_LR_3", snapshot_date_str="2024-08-01", fallback_dir="/app/airflow/model_bank/"):
    """
    Load model from MLflow Registry or fallback to pickle file.

    Args:
        model_name: Name of the model in MLflow Registry
        snapshot_date: Snapshot date string (e.g., "2024-08-01") for pickle fallback
        fallback_dir: Directory containing pickle files

    Returns:
        Loaded model object
    """
    snapshot_date = snapshot_date_str.replace("-", "_")

    # Try loading from MLflow Registry
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        print(f"🔍 Attempting to load model from MLflow Registry: {model_name}")

        model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        print(f"✅ Successfully loaded model from MLflow Registry")

        # Try to load scaler artifact if needed
        try:
            client = MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest_version = versions[0]
                run_id = latest_version.run_id
                scaler_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, 
                    artifact_path="preprocessing/temp_scaler.pkl"
                )
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✅ Successfully loaded scaler from MLflow")
                return model, scaler
        except Exception as e:
            print(f"⚠️  Could not load scaler from MLflow: {e}")
            return model, None

        return model, None

    except Exception as e:
        print(f"❌ Failed to load from MLflow Registry: {e}")

        # Fallback to pickle file
        pickle_filename = f"credit_model_LR_{snapshot_date}.pkl"
        pickle_path = f"{fallback_dir}{pickle_filename}"

        try:
            print(f"🔍 Attempting to load from pickle: {pickle_path}")
            with open(pickle_path, 'rb') as f:
                model_data = pickle.load(f)

            # Handle different pickle formats
            if isinstance(model_data, dict):
                model = model_data.get('model')
                scaler = model_data.get('scaler')
                print(f"✅ Successfully loaded model and scaler from pickle (dict format)")
            elif isinstance(model_data, tuple):
                model, scaler = model_data
                print(f"✅ Successfully loaded model and scaler from pickle (tuple format)")
            else:
                model = model_data
                scaler = None
                print(f"✅ Successfully loaded model from pickle (single object)")

            return model, scaler

        except FileNotFoundError:
            print(f"❌ Pickle file not found: {pickle_path}")
            raise Exception(f"Could not load model from MLflow or pickle file: {pickle_path}")
        except Exception as e:
            print(f"❌ Failed to load from pickle: {e}")
            raise


if __name__ == "__main__":
    # Try to load model
    model, scaler = load_model(
        model_name="credit_model_LR_3",
        snapshot_date_str="2024-08-01",
        fallback_dir="/app/airflow/model_bank/"
    )

    print(f"\nModel type: {type(model).__name__}")
    if scaler:
        print(f"Scaler type: {type(scaler).__name__}")

#---------------------------------------------------
# LOAD FEATURE STORE
#---------------------------------------------------

FEATURE_DIR = "/app/datamart/gold/feature_store"
features_store_sdf = spark.read.parquet(FEATURE_DIR)
print("row_count:",features_store_sdf.count())

features_store_sdf.show(1)

for snapshot_date in ['2024-07-01' ,'2024-08-01', '2024-09-01', '2024-10-01', '2024-11-01', '2024-12-01', '2025-01-01']:
    
    config["snapshot_date_str"] = snapshot_date
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    
    try:
        features_sdf = features_store_sdf.filter((col("snapshot_date") == config["snapshot_date"]))
    except Exception as e:
        print(f"⚠️ Using application_date instead of snapshot_date due to: {e}")
        features_sdf = features_store_sdf.filter((col("application_date") == config["snapshot_date"]))
    
    print("extracted features_sdf", features_sdf.count(), config["snapshot_date"])
    
    features_pdf = features_sdf.toPandas()
    
    #---------------------------------------------------
    # PREPROCESS DATA
    #---------------------------------------------------
    
    # prepare X_inference
    feature_cols = [fe_col for fe_col in features_pdf.columns if fe_col not in ['Customer_ID', 'application_date', 'snapshot_date']]
    X_features = features_pdf[feature_cols]
    
    X_features.head()
    
    # Apply scaler if it exists
    if scaler is not None:
        print("🔄 Applying scaler transformation...")
        X_scaled = scaler.transform(X_features)
        print(f"✅ Features scaled: {X_features.shape} -> {X_scaled.shape}")
    else:
        print("⚠️  No scaler found, using raw")
        # --- Scaling ---
        X_scaled = X_features
    
    #---------------------------------------------------
    # MAKE INFERENCE & PREPARE OUTPUT
    #---------------------------------------------------
    
    y_inference = model.predict_proba(X_scaled)[:, 1]
    y_inference
    
    # prepare output
    y_inference_pdf = features_pdf[["Customer_ID","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    
    #---------------------------------------------------
    # SAVE TO GOLD LAYER
    #---------------------------------------------------
    
    # create gold datalake
    snapshot_date_path = config["snapshot_date_str"].replace('-','_')
    model_naming = config["model_name"]
    prediction_directory = f"/app/datamart/gold/model_predictions/{model_naming}/"
    print(prediction_directory)
    
    if not os.path.exists(prediction_directory):
        os.makedirs(prediction_directory)
    
    filepath = os.path.join(prediction_directory, f"predictions_{snapshot_date_path}.csv")
    y_inference_pdf.to_csv(filepath, index=False)

spark.stop()

print('\n\n---completed job---\n\n')

#---------------------------------------------------
# END
#---------------------------------------------------
