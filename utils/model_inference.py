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

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.xgboost

# to call this script: python 05_model_inference.py --snapshotdate "2024-09-01"

def load_model(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    
    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = "/app/airflow/model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)

    #---------------------------------------------------
    # LOAD MODEL
    #---------------------------------------------------
    
    # --- load model artefact from model bank or MLflow ---
    model_artefact = None
    model = None
    transformer_stdscaler = None
    
    # Try loading from MLflow first
    try:
        # Extract run name from model name (assumes format: credit_model_YYYY_MM_DD.pkl)
        date_part = config["model_name"].replace("credit_model_LR_2", "").replace(".pkl", "").replace("_", "-")
        run_name = f"run_{date_part}"
        
        # Search for the run
        experiment = mlflow.get_experiment_by_name("credit_model_LR_2")
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'"
            )
            
            if not runs.empty:
                run_id = runs.iloc[0]['run_id']
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.xgboost.load_model(model_uri)
                
                # Load scaler from pickle (MLflow doesn't store preprocessing transformers)
                with open(config["model_artefact_filepath"], 'rb') as file:
                    model_artefact = pickle.load(file)
                transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
                
                print(f"Model loaded from MLflow! Run ID: {run_id}")
            else:
                raise Exception("Run not found in MLflow")
        else:
            raise Exception("Experiment not found in MLflow")
            
    except Exception as e:
        print(f"MLflow loading failed: {str(e)}")
        print("Falling back to pickle file...")
        
        # Fall back to pickle file
        with open(config["model_artefact_filepath"], 'rb') as file:
            model_artefact = pickle.load(file)
        
        model = model_artefact["model"]
        transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]
        
        print("Model loaded from pickle file! " + config["model_artefact_filepath"])
