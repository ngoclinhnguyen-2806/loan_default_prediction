import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import sys
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
import data_processing_bronze_table

# Initialize SparkSession (local dev)
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Reduce Spark logs to errors
spark.sparkContext.setLogLevel("ERROR")

# === Bronze layer directories ===
bronze_lms_directory = "/app/datamart/bronze/lms_loan_daily/"
bronze_clickstream_directory = "/app/datamart/bronze/feature_clickstream/"
bronze_attr_directory = "/app/datamart/bronze/features_attributes/"
bronze_fin_directory = "/app/datamart/bronze/features_financials/"

# Ensure Bronze dirs exist
for directory in [
    bronze_lms_directory,
    bronze_clickstream_directory,
    bronze_attr_directory,
    bronze_fin_directory
]:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory (if missing): {directory}")

# Ingest raw CSVs -> Bronze tables in CSV (split by snapshot date)
data_processing_bronze_table.process_bronze_table("/app/data/lms_loan_daily.csv",bronze_lms_directory, spark)
data_processing_bronze_table.process_bronze_table("/app/data/feature_clickstream.csv",bronze_clickstream_directory, spark)
data_processing_bronze_table.process_bronze_table("/app/data/features_attributes.csv",bronze_attr_directory, spark)
data_processing_bronze_table.process_bronze_table("/app/data/features_financials.csv",bronze_fin_directory, spark)