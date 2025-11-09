# %%
import os
import glob
import numpy as np
import pyspark
import pyspark.sql.functions as F

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
print("predictions:")

# Load Gold Features
gold_label_directory = "/app/datamart/gold/label_store/"
files_list = [gold_label_directory+os.path.basename(f) for f in glob.glob(os.path.join(gold_label_directory, '*'))]
label_df = spark.read.option("header", "true").parquet(*files_list)
print("True labels:")

# load features (same snapshot granularity as predictions)
features_dir = "/app/datamart/gold/feature_store/"
files = [features_dir + os.path.basename(f) for f in glob.glob(os.path.join(features_dir, '*'))]
df_features = spark.read.parquet(*files)
print("Features:")

# %%
label_df.groupBy("snapshot_date") \
        .agg(F.count("*").alias("count")) \
        .orderBy("snapshot_date") \
        .show(truncate=False)

# %%
merged_df = df_predictions.alias("p").join(
    df_features.alias("x"),
    on=["Customer_ID","snapshot_date"],
    how="inner"
).join(
    label_df.select("Customer_ID","snapshot_date","label").alias("y"),
    on=["Customer_ID","snapshot_date"],
    how="inner"   # labels may be missing
)

# %% [markdown]
# # Data Drift (feature distribution shift)
# 
# compare feature distributions now vs a baseline snapshot
# 
# Kolmogorov–Smirnov test per numeric feature
# 
# p < 0.05 → feature has drifted
# 
# KS stat closer to 1 → heavy drift

from scipy.stats import ks_2samp

baseline = merged_df.filter(F.col("snapshot_date") == "2024-11-01").toPandas()
current  = merged_df.filter(F.col("snapshot_date") == "2024-12-01").toPandas()

for col in ["Annual_Income","Num_of_Loan","Credit_History_Age","fe_1","fe_2","fe_3"]:
    stat, p = ks_2samp(baseline[col].dropna(), current[col].dropna())
    print(col, "KS stat:", stat, "p-value:", p)



# %% [markdown]
# # Concept Drift (model behavior or learned relationship changed)

# %%
df = merged_df.withColumn("prediction_error", F.abs(F.col("label") - F.col("model_predictions")))

# %%
import numpy as np

def psi(expected, actual, bins=10):
    e_perc, _ = np.histogram(expected, bins=bins)
    a_perc, _ = np.histogram(actual, bins=bins)
    e_perc = e_perc/len(expected)
    a_perc = a_perc/len(actual)
    psi_val = sum((e - a) * np.log(e/a) for e, a in zip(e_perc, a_perc) if e!=0 and a!=0)
    return psi_val

base = baseline["model_predictions"]
curr = current["model_predictions"]
print("PSI:", psi(base, curr))