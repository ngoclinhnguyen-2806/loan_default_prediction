"""
CLI:
  python /app/scripts/03_create_features.py                # latest application_date
  python /app/scripts/03_create_features.py all            # all dates
  python /app/scripts/03_create_features.py range 2025-01-01 2025-03-31
  python /app/scripts/03_create_features.py 2025-11-01     # specific date
"""

import sys
from pyspark.sql import SparkSession, functions as F
from pathlib import Path

# Make utils importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from gold_features_processing_optimized import build_features

FEATURE_STORE_PATH = "/app/datamart/gold/feature_store"
SILVER_DIR = "/app/datamart/silver"
APP_STORE_PATH = "/app/datamart/gold/application_store"   # if your folder is 'application_storage', change here

def main(mode="latest", start_date=None, end_date=None):
    spark = SparkSession.builder.appName("CreateFeatures").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # ✅ only overwrite the partition being written, keep others
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    # Load application store and enumerate dates
    apps_all = spark.read.parquet(APP_STORE_PATH).withColumn("application_date", F.to_date("application_date"))
    available_dates = [
        r["application_date"].strftime("%Y-%m-%d")
        for r in apps_all.select("application_date").distinct().orderBy("application_date").collect()
    ]
    if not available_dates:
        print("⚠️ No application_date found in application store.")
        spark.stop(); return

    if mode == "latest":
        dates_to_run = [available_dates[-1]]
    elif mode == "all":
        dates_to_run = available_dates
    elif mode == "range":
        if not start_date or not end_date:
            print("❌ Usage: python 03_create_features.py range <start_date> <end_date>")
            spark.stop(); return
        dates_to_run = [d for d in available_dates if start_date <= d <= end_date]
        if not dates_to_run:
            print(f"⚠️ No available dates between {start_date} and {end_date}.")
            spark.stop(); return
    else:
        dates_to_run = [mode]  # specific date

    print(f"🗓 Running feature generation for application dates: {dates_to_run}")

    for app_date in dates_to_run:
        apps = apps_all.filter(F.col("application_date") == F.lit(app_date))

        # build features
        features = build_features(apps, SILVER_DIR, app_date, spark)

        # ensure application_date exists & is date typed for partitioning
        features = features.withColumn("application_date", F.to_date("application_date"))

        # ✅ dynamic overwrite by partition keeps previous dates
        (features
            .write
            .mode("overwrite")
            .partitionBy("application_date")
            .parquet(FEATURE_STORE_PATH))

        print(f"✅ Features for application_date={app_date} → {FEATURE_STORE_PATH}")

    spark.stop()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        main("latest")
    elif args[0] == "all":
        main("all")
    elif args[0] == "range":
        if len(args) < 3:
            print("❌ Usage: python 03_create_features.py range <start_date> <end_date>")
        else:
            main("range", args[1], args[2])
    else:
        main(args[0])
