"""
CLI examples:
  python /app/scripts/03_create_applications.py
  python /app/scripts/03_create_applications.py all
  python /app/scripts/03_create_applications.py range 2025-01-01 2025-03-31
  python /app/scripts/03_create_applications.py 2025-11-01
"""

import sys
from pyspark.sql import SparkSession, functions as F

def build_apps(spark, dates_to_run):
    lms_path = "/app/data/lms_loan_daily.csv"
    output_path = "/app/datamart/gold/application_store"

    # Load LMS CSV
    lms = (
        spark.read.csv(lms_path, header=True, inferSchema=True)
             .withColumn("loan_start_date", F.to_date("loan_start_date"))
             .select("Customer_ID", "loan_start_date")
             .dropna(subset=["loan_start_date"])
             .dropDuplicates(["Customer_ID", "loan_start_date"])
    )

    # Filter if specific dates provided
    if dates_to_run:
        lms = lms.filter(F.col("loan_start_date").isin([F.lit(d) for d in dates_to_run]))

    # Rename + add static columns
    apps = (
        lms.withColumnRenamed("loan_start_date", "application_date")
           .withColumn("loan_amt", F.lit(10000))
           .withColumn("tenure", F.lit(10))
           .withColumn("application_date", F.col("application_date").cast("date"))
           .select("Customer_ID", "application_date", "loan_amt", "tenure")
    )

    # ✅ Ensure application_date is both a column and partition key
    (
        apps
        .write
        .mode("overwrite")
        .partitionBy("application_date")
        .option("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .parquet(output_path)
    )

    return output_path


def main():
    
    spark = SparkSession.builder.appName("CreateApplications").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Load CSV to find available dates
    lms = (
        spark.read.csv("/app/data/lms_loan_daily.csv", header=True, inferSchema=True)
             .withColumn("loan_start_date", F.to_date("loan_start_date"))
             .dropna(subset=["loan_start_date"])
    )

    available_dates = [
        r["loan_start_date"].strftime("%Y-%m-%d")
        for r in lms.select("loan_start_date").distinct().orderBy("loan_start_date").collect()
    ]

    if not available_dates:
        print("❌ No valid loan_start_date found in /app/data/lms_loan_daily.csv")
        spark.stop()
        sys.exit(1)

    args = sys.argv[1:]
    dates_to_run = None

    if not args:
        dates_to_run = [available_dates[-1]]  # latest
    elif args[0] == "all":
        dates_to_run = available_dates
    elif args[0] == "range":
        if len(args) < 3:
            print("❌ Usage: 03_create_applications.py range <start_date> <end_date>")
            spark.stop()
            sys.exit(1)
        start_date, end_date = args[1], args[2]
        dates_to_run = [d for d in available_dates if start_date <= d <= end_date]
    else:
        dates_to_run = [args[0]]  # specific date

    output_path = build_apps(spark, dates_to_run)
    print(f"✅ Application store written for dates={dates_to_run} → {output_path}")

    spark.stop()


if __name__ == "__main__":
    main()
