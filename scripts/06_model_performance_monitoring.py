# %% [markdown]
# # Model performance monitoring

# %%
import os
import glob
import random
import pprint

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyspark

# %%
# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Read the predictions parquet file
predictions_path = "/app/datamart/gold/model_predictions/credit_model_2024_10_01/credit_model_2024_10_01_predictions_2024_11_01.parquet"
df_predictions = spark.read.parquet(predictions_path)

# %%
# Load true labels and join
gold_label_directory = "/app/datamart/gold/label_store/"
files_list = [gold_label_directory+os.path.basename(f) for f in glob.glob(os.path.join(gold_label_directory, '*'))]
label_df = spark.read.option("header", "true").parquet(*files_list)

merged_df = df_predictions.join(
    label_df,
    on=["snapshot_date", "Customer_ID"],
    how="inner"
)
# %%
from pyspark.sql import functions as F

# filter model's prediction
model_to_eval = "credit_model_2024_10_01.pkl"
df_eval = merged_df.filter(F.col("model_name") == model_to_eval)
pdf = df_eval.select("label", "model_predictions").toPandas()

y_true = pdf["label"]
y_pred = pdf["model_predictions"]
y_pred_binary = (y_pred >= 0.5).astype(int)  # adjust threshold if needed

# %%
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss

metrics = {
    "AUC": roc_auc_score(y_true, y_pred),
    "F1": f1_score(y_true, y_pred_binary),
    "Precision": precision_score(y_true, y_pred_binary),
    "Recall": recall_score(y_true, y_pred_binary),
    "Brier": brier_score_loss(y_true, y_pred),
    "Pred_Mean": y_pred.mean(),
    "Positive_Rate": y_pred_binary.mean()
}

import pandas as pd

df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
print("\n📊 Model Metrics Table")
print(df_metrics.to_string(index=False, float_format="%.4f"))
