"""
Gold Layer Processing Script
Run this to process Silver → Gold (Feature Store + Label Store) independently

Usage:
    python 03_run_gold_processing.py
    python 03_run_gold_processing.py --start-date 2023-01-01 --end-date 2024-12-01
    python 03_run_gold_processing.py --single-date 2023-06-01
    python 03_run_gold_processing.py --features-only
    python 03_run_gold_processing.py --labels-only
"""

import os
import glob
import argparse
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pathlib import Path
import sys

# Add /app/utils to PYTHONPATH automatically
sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))

import gold_features_processing
import gold_label_processing

SILVER_DIR = '/app/datamart/silver/'
GOLD_FEATURE_DIR = '/app/datamart/gold/feature_store/'
GOLD_LABEL_DIR = '/app/datamart/gold/label_store/'


def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


def verify_silver_tables(silver_directory, table_names=None):
    if table_names is None:
        table_names = ['loan_daily', 'feature_clickstream', 
                      'features_attributes', 'features_financials']
    
    print("\n" + "="*80)
    print("VERIFYING SILVER LAYER TABLES")
    print("="*80)
    
    status = {}
    all_available = True
    
    for table_name in table_names:
        table_path = os.path.join(silver_directory, table_name)
        
        if os.path.exists(table_path):
            parquet_files = glob.glob(os.path.join(table_path, "*.parquet"))
            if parquet_files:
                status[table_name] = {
                    'exists': True,
                    'partition_count': len(parquet_files),
                    'path': table_path
                }
                print(f"✓ {table_name}: {len(parquet_files)} partitions found")
            else:
                status[table_name] = {
                    'exists': False,
                    'partition_count': 0,
                    'path': table_path
                }
                print(f"✗ {table_name}: Directory exists but no parquet files")
                all_available = False
        else:
            status[table_name] = {
                'exists': False,
                'partition_count': 0,
                'path': table_path
            }
            print(f"✗ {table_name}: Directory not found")
            all_available = False
    
    print("="*80)
    
    if not all_available:
        print("\n⚠ WARNING: Some silver tables are missing!")
        print("Run the full pipeline or bronze→silver processing first.")
    
    return status, all_available


def verify_gold_output(gold_feature_directory, gold_label_directory, spark):
    print("\n" + "="*80)
    print("GOLD LAYER OUTPUT SUMMARY")
    print("="*80)
    
    print("\n--- FEATURE STORE ---")
    feature_parquet_files = glob.glob(os.path.join(gold_feature_directory, "*.parquet"))
    
    if not feature_parquet_files:
        print("No parquet files found in feature store")
    else:
        print(f"Total feature partitions: {len(feature_parquet_files)}")
    
    print("\n--- LABEL STORE ---")
    label_parquet_files = glob.glob(os.path.join(gold_label_directory, "*.parquet"))
    
    if not label_parquet_files:
        print("No parquet files found in label store")
    else:
        print(f"Total label partitions: {len(label_parquet_files)}")
    
    if not feature_parquet_files or not label_parquet_files:
        return
    
    try:
        features_df = spark.read.parquet(*feature_parquet_files)
        
        print(f"\n{'='*80}")
        print("FEATURE STORE STATISTICS")
        print("="*80)
        print(f"Total applications: {features_df.count():,}")
        print(f"Total features: {len(features_df.columns)}")
        print(f"Unique customers: {features_df.select('Customer_ID').distinct().count():,}")
        
        date_stats = features_df.agg(
            F.min("application_date").alias("min_date"),
            F.max("application_date").alias("max_date")
        ).collect()[0]
        print(f"Application date range: {date_stats['min_date']} to {date_stats['max_date']}")
        
    except Exception as e:
        print(f"\n✗ Error analyzing feature store: {str(e)}")
    
    try:
        labels_df = spark.read.parquet(*label_parquet_files)
        
        print(f"\n{'='*80}")
        print("LABEL STORE STATISTICS")
        print("="*80)
        print(f"Total labels: {labels_df.count():,}")
        
        if "default_label" in labels_df.columns:
            default_count = labels_df.filter(col("default_label") == 1).count()
            total_count = labels_df.count()
            default_rate = (default_count / total_count * 100) if total_count > 0 else 0
            print(f"Default rate: {default_rate:.2f}% ({default_count:,}/{total_count:,})")
        
    except Exception as e:
        print(f"\n✗ Error analyzing label store: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Process Silver → Gold layer')
    parser.add_argument('--start-date', type=str, default='2023-01-01')
    parser.add_argument('--end-date', type=str, default='2024-12-01')
    parser.add_argument('--single-date', type=str, default=None)
    parser.add_argument('--skip-verification', action='store_true')
    parser.add_argument('--features-only', action='store_true')
    parser.add_argument('--labels-only', action='store_true')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GOLD LAYER PROCESSING - CONFIGURATION")
    print("="*80)
    print(f"Silver directory: {SILVER_DIR}")
    print(f"Gold feature directory: {GOLD_FEATURE_DIR}")
    print(f"Gold label directory: {GOLD_LABEL_DIR}")
    
    if args.single_date:
        dates_to_process = [args.single_date]
        print(f"Mode: Single date ({args.single_date})")
    else:
        dates_to_process = generate_first_of_month_dates(args.start_date, args.end_date)
        print(f"Mode: Batch ({args.start_date} to {args.end_date}, {len(dates_to_process)} dates)")
    
    if args.features_only:
        print("Processing: Features only")
    elif args.labels_only:
        print("Processing: Labels only")
    else:
        print("Processing: Both features and labels")
    
    print("="*80)
    
    print("\nInitializing Spark...")
    spark = (pyspark.sql.SparkSession.builder
        .appName("gold_processing")
        .master("local[*]")
        .config("spark.driver.memory", "24g")
        .config("spark.driver.maxResultSize", "8g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate())
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark initialized")
    
    if not args.skip_verification:
        status, all_available = verify_silver_tables(SILVER_DIR)
        if not all_available:
            response = input("\nProceed anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    
    for directory in [GOLD_FEATURE_DIR, GOLD_LABEL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
    
    print("\n" + "="*80)
    print("PROCESSING GOLD LAYER")
    print("="*80)
    
    success_count = 0
    failed_dates = []
    
    for idx, date_str in enumerate(dates_to_process, 1):
        print(f"\n[{idx}/{len(dates_to_process)}] Processing {date_str}...")
        
        try:
            if not args.labels_only:
                gold_features_processing.process_gold_features(
                    date_str, SILVER_DIR, GOLD_FEATURE_DIR, spark)
                print(f"✓ Features completed for {date_str}")
            
            if not args.features_only:
                gold_label_processing.process_gold_labels(
                    date_str, SILVER_DIR, GOLD_LABEL_DIR, spark)
                print(f"✓ Labels completed for {date_str}")
            
            success_count += 1
            
        except Exception as e:
            print(f"✗ Failed {date_str}: {str(e)}")
            failed_dates.append((date_str, str(e)))
    
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total dates: {len(dates_to_process)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_dates)}")
    
    if failed_dates:
        print("\nFailed dates:")
        for date_str, error in failed_dates:
            print(f"  - {date_str}: {error}")
    
    verify_gold_output(GOLD_FEATURE_DIR, GOLD_LABEL_DIR, spark)
    
    print("\n" + "="*80)
    print("GOLD PROCESSING COMPLETED")
    print("="*80)
    
    spark.stop()


if __name__ == "__main__":
    main()
