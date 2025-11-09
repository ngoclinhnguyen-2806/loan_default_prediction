"""
CLI Usage Examples:
  python /app/scripts/04_create_gold_labels.py all
  python /app/scripts/04_create_gold_labels.py 2025-01-01
  python /app/scripts/04_create_gold_labels.py range 2025-01-01 2025-03-31
"""

import os
import sys
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
SILVER_DIR = "/app/datamart/silver/lms_loan_daily"
GOLD_LABEL_DIR = "/app/datamart/gold/label_store"
DPD_THRESHOLD = 30
MOB_THRESHOLD = 6

# -------------------------------------------------------------------------
# CORE LOGIC
# -------------------------------------------------------------------------
def process_labels_for_snapshot(spark, snapshot_date_str, dpd=DPD_THRESHOLD, mob=MOB_THRESHOLD):
    """Create labels for a specific snapshot_date and save to Gold label store."""
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    print(f"\n=== Processing snapshot_date: {snapshot_date_str} ===")

    if not os.path.exists(SILVER_DIR):
        raise FileNotFoundError(f"❌ Silver directory not found: {SILVER_DIR}")

    # 1️⃣ Load Silver data
    df = spark.read.parquet(SILVER_DIR)
    df = df.withColumn("snapshot_date", col("snapshot_date").cast("date"))

    # 2️⃣ Filter rows for this snapshot date
    df = df.filter(col("snapshot_date") == F.lit(snapshot_date))
    row_count = df.count()
    print(f"Loaded {row_count} rows for snapshot_date {snapshot_date_str}")
    if row_count == 0:
        print(f"⚠️ No data found for {snapshot_date_str}, skipping.")
        return None

    # 3️⃣ Filter loans at given MOB and compute label
    df = df.filter(col("mob") == mob)
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))

    # 4️⃣ Keep relevant columns
    keep_cols = ["loan_id", "Customer_ID", "label", "label_def", "snapshot_date"]
    df = df.select(*[c for c in keep_cols if c in df.columns])

    # 5️⃣ Save to Gold
    os.makedirs(GOLD_LABEL_DIR, exist_ok=True)
    out_file = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
    out_path = os.path.join(GOLD_LABEL_DIR, out_file)

    df.write.mode("overwrite").parquet(out_path)
    print(f"✅ Saved labels to {out_path} ({df.count()} rows)")

    return df

# -------------------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------------------
def main():
    spark = (
        pyspark.sql.SparkSession.builder
        .appName("gold_label_creation")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    args = sys.argv[1:]

    # Load all available snapshot dates
    df_all = spark.read.parquet(SILVER_DIR)
    dates = sorted([
        r[0].strftime("%Y-%m-%d")
        for r in df_all.select("snapshot_date").distinct().orderBy("snapshot_date").collect()
    ])

    if not args or args[0].lower() == "all":
        print(f"\n=== Running label creation for ALL {len(dates)} snapshot dates ===")
        for d in dates:
            process_labels_for_snapshot(spark, d)

    elif args[0].lower() == "range" and len(args) == 3:
        start, end = args[1], args[2]
        date_range = [d for d in dates if start <= d <= end]
        if not date_range:
            print(f"⚠️ No snapshot dates between {start} and {end}")
        else:
            print(f"\n=== Running label creation for range {start} → {end} ===")
            for d in date_range:
                process_labels_for_snapshot(spark, d)

    else:
        snapshot_date_str = args[0]
        if snapshot_date_str not in dates:
            print(f"⚠️ Snapshot date {snapshot_date_str} not found in Silver data. Processing anyway.")
        process_labels_for_snapshot(spark, snapshot_date_str)

    spark.stop()
    print("\n--- Gold Label Creation Completed ---\n")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
