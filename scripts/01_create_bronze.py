"""
Call this script: python 01_create_bronze.py
"""

import os
import pyspark
from pyspark.sql.functions import year, month, to_date

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
SOURCE_FILES = {
    "lms_loan_daily": "data/lms_loan_daily.csv",
    "feature_clickstream": "data/feature_clickstream.csv",
    "features_attributes": "data/features_attributes.csv",
    "features_financials": "data/features_financials.csv",
}
BRONZE_DIR = "/app/datamart/bronze"

# -------------------------------------------------------------------------
# PROCESSING FUNCTION
# -------------------------------------------------------------------------
def process_bronze_table(spark, table):
    csv_path = SOURCE_FILES.get(table)
    if not csv_path or not os.path.exists(csv_path):
        print(f"⚠️  Missing source for {table}")
        return

    print(f"Processing {table} ...")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    if "snapshot_date" not in df.columns:
        print(f"❌ {table}: snapshot_date column missing — skipped.")
        return

    # Ensure snapshot_date is proper date type
    df = df.withColumn("snapshot_date", to_date("snapshot_date"))

    # Add year and month for partitioning
    df = df.withColumn("year", year("snapshot_date")).withColumn("month", month("snapshot_date"))

    out_dir = os.path.join(BRONZE_DIR, table)
    os.makedirs(out_dir, exist_ok=True)

    df.write.mode("overwrite").partitionBy("year", "month").parquet(out_dir)
    print(f"✅ {table}: written to {out_dir}")

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    print("\n--- Starting Bronze Layer Processing ---\n")
    spark = pyspark.sql.SparkSession.builder.appName("bronze_processing").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    os.makedirs(BRONZE_DIR, exist_ok=True)

    for table in SOURCE_FILES.keys():
        try:
            process_bronze_table(spark, table)
        except Exception as e:
            print(f"❌ Error processing {table}: {e}")

    spark.stop()
    print("\n--- Bronze Layer Processing Completed ---\n")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
