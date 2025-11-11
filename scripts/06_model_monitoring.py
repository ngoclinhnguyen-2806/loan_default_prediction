import os
import glob
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import ks_2samp

#---------------------------------------------------
# HELPER FUNCTIONS
#---------------------------------------------------

def calculate_gini(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1

def calculate_ks(y_true, y_pred):
    scores_pos = y_pred[y_true == 1]
    scores_neg = y_pred[y_true == 0]
    return ks_2samp(scores_pos, scores_neg).statistic

#---------------------------------------------------
# INITIALIZE SPARK
#---------------------------------------------------
spark = pyspark.sql.SparkSession.builder \
    .appName("ModelMonitoringBuilder") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


#---------------------------------------------------
# GET PREDICTIONS
#---------------------------------------------------

prediction_dir = "/app/datamart/gold/model_predictions/LR/"
inference_data = None

for snapshot_date_str in ['2024-07-01' ,'2024-08-01', '2024-09-01', '2024-10-01', '2024-11-01', '2024-12-01', '2025-01-01']:
    partition_name = f"predictions_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = prediction_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"✅ loaded from: {filepath}, rows: {df.count()}")

    if inference_data is None:
        inference_data = df
    else:
        inference_data = inference_data.unionByName(df)

print("📊 Total rows after append:", inference_data.count())

inference_data.show(5)

max_date = inference_data.agg(F.max("snapshot_date").alias("max_date")).collect()[0]["max_date"]
min_date = inference_data.agg(F.min("snapshot_date").alias("min_date")).collect()[0]["min_date"]

inference_data = inference_data.select('Customer_ID', 'snapshot_date', 'model_predictions')
inference_data = inference_data.toPandas()

# Convert snapshot_date to datetime if it's not already
inference_data['snapshot_date'] = pd.to_datetime(inference_data['snapshot_date'])

# Check unique months available
print("Available months:")
print(inference_data['snapshot_date'].dt.to_period('M').unique())

inference_data.head()

#---------------------------------------------------
# GET LABELS
#---------------------------------------------------

folder_path = "/app/datamart/gold/label_store/"
files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
labels_df = spark.read.option("header", "true").parquet(*files_list)
print("row_count:",labels_df.count())

labels_df = labels_df.select('Customer_ID', 'snapshot_date', 'label') \
                     .filter((col("snapshot_date") >= min_date) & (col("snapshot_date") <= max_date))
labels_df.show(10)
labels_df = labels_df.toPandas()

# Convert snapshot_date to datetime if it's not already
labels_df['snapshot_date'] = pd.to_datetime(labels_df['snapshot_date'])
labels_df.head()

spark.stop()

#---------------------------------------------------
# MERGE DATA
#---------------------------------------------------

inference_data = labels_df.merge(
    inference_data, 
    on=['Customer_ID'], 
    how='inner'
)
inference_data.head()

#---------------------------------------------------
# MONITORING
#---------------------------------------------------

# Loop through each monitoring month
monitoring_results = []

for month in ['2024-09-01', '2024-10-01', '2024-11-01', '2024-12-01', '2025-01-01']:
    # Filter data for this month
    month_data = inference_data[inference_data['snapshot_date_x'] == month]

    # Skip if no data for this month
    if len(month_data) == 0:
        print(f"Warning: No data found for {month}")
        continue

    y_true = month_data['label']
    y_pred = month_data['model_predictions']

    metrics = {
        "month": month,
        "n_samples": len(y_true),
        "AUC": round(roc_auc_score(y_true, y_pred), 4),
        "Gini": round(calculate_gini(y_true, y_pred), 4),
        "KS": round(calculate_ks(y_true, y_pred), 4),
        "Brier": round(brier_score_loss(y_true, y_pred), 4),
        "Pred_Mean": round(y_pred.mean(), 4),
        "Actual_Default_Rate": round(y_true.mean(), 4),
        "Default_Rate_Diff": round(abs(y_pred.mean() - y_true.mean()), 4)
    }

    monitoring_results.append(metrics)

# Convert to DataFrame
monitoring_df = pd.DataFrame(monitoring_results)
print("\n=== Monthly Monitoring Results ===")
print(monitoring_df.to_string(index=False))

#---------------------------------------------------
# WRITE MONITORING RESULT TO CSV
#---------------------------------------------------

monitoring_directory = f"/app/datamart/gold/model_monitoring/"

if not os.path.exists(monitoring_directory):
    os.makedirs(monitoring_directory)

filepath = os.path.join(monitoring_directory, f"predictions_{min_date}_{max_date}.csv")
monitoring_df.to_csv(filepath, index=False)

