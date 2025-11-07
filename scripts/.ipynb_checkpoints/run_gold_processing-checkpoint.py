"""
Gold Layer Processing Script
Run this to process Silver → Gold (Feature Store) independently

Usage:
    python run_gold_processing.py
    python run_gold_processing.py --start-date 2023-01-01 --end-date 2024-12-01
    python run_gold_processing.py --single-date 2023-06-01
"""

import os
import glob
import argparse
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

import utils.data_processing_gold_table


def generate_first_of_month_dates(start_date_str, end_date_str):
    """
    Generate list of first-of-month dates between start and end dates
    
    Args:
        start_date_str: Start date string 'YYYY-MM-DD'
        end_date_str: End date string 'YYYY-MM-DD'
    
    Returns:
        List of date strings
    """
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
    """
    Verify that required silver tables exist
    
    Args:
        silver_directory: Base silver directory
        table_names: List of table names to check (default: all 4 tables)
    
    Returns:
        dict: Status of each table
    """
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


def verify_gold_output(gold_directory, spark):
    """
    Verify and summarize gold layer output
    
    Args:
        gold_directory: Gold feature store directory
        spark: SparkSession
    """
    print("\n" + "="*80)
    print("GOLD LAYER OUTPUT SUMMARY")
    print("="*80)
    
    parquet_files = glob.glob(os.path.join(gold_directory, "*.parquet"))
    
    if not parquet_files:
        print("No parquet files found in gold layer")
        return
    
    print(f"\nTotal partitions created: {len(parquet_files)}")
    print(f"\nPartitions:")
    for f in sorted(parquet_files):
        print(f"  - {os.path.basename(f)}")
    
    # Load and analyze
    try:
        df = spark.read.parquet(*parquet_files)
        
        print(f"\n{'='*80}")
        print("FEATURE STORE STATISTICS")
        print("="*80)
        print(f"Total applications: {df.count():,}")
        print(f"Total features: {len(df.columns)}")
        print(f"Unique customers: {df.select('Customer_ID').distinct().count():,}")
        
        # Date range
        date_stats = df.agg(
            F.min("application_date").alias("min_date"),
            F.max("application_date").alias("max_date")
        ).collect()[0]
        print(f"Application date range: {date_stats['min_date']} to {date_stats['max_date']}")
        
        # Default rate
        if "default_label" in df.columns:
            default_count = df.filter(col("default_label") == 1).count()
            total_count = df.count()
            default_rate = (default_count / total_count * 100) if total_count > 0 else 0
            print(f"\nDefault rate: {default_rate:.2f}% ({default_count:,}/{total_count:,})")
        
        # Feature categories
        print(f"\n{'='*80}")
        print("FEATURE CATEGORIES")
        print("="*80)
        
        feature_categories = {
            'Identity': ['loan_id', 'Customer_ID'],
            'Dates': ['application_date', 'snapshot_date'],
            'Loan Info': ['loan_amt', 'tenure', 'requested_amount', 'requested_tenure'],
            'Capacity': ['DTI', 'log_Annual_Income', 'income_band', 'Annual_Income', 'Outstanding_Debt'],
            'Credit Depth': ['Credit_History_Age_Year', 'Credit_History_Age_Month', 'Num_of_Loan_active'],
            'Delinquency': ['Num_of_Delayed_Payment_3m', 'Num_of_Delayed_Payment_6m', 
                          'Num_of_Delayed_Payment_12m', 'ever_30dpd_prior', 'max_dpd_prior'],
            'Demographics': ['Age', 'age_band', 'Occupation'],
            'Application': ['estimated_EMI', 'EMI_to_income'],
            'Labels': ['default_label']
        }
        
        for category, features in feature_categories.items():
            available = [f for f in features if f in df.columns]
            if available:
                print(f"\n{category}: {len(available)} features")
                for f in available[:5]:  # Show first 5
                    print(f"  - {f}")
                if len(available) > 5:
                    print(f"  ... and {len(available) - 5} more")
        
        # Clickstream features
        clickstream_features = [c for c in df.columns if c.startswith('fe_')]
        if clickstream_features:
            print(f"\nClickstream: {len(clickstream_features)} features")
            print(f"  - {clickstream_features[0]} ... {clickstream_features[-1]}")
        
        # Behavioral features
        payment_features = [c for c in df.columns if 'Payment_Behaviour' in c]
        credit_mix_features = [c for c in df.columns if 'Credit_Mix' in c]
        loan_type_features = [c for c in df.columns if 'Type_of_Loan' in c]
        
        if payment_features or credit_mix_features or loan_type_features:
            print(f"\nBehavioral (one-hot encoded):")
            if payment_features:
                print(f"  - Payment Behaviour: {len(payment_features)} categories")
            if credit_mix_features:
                print(f"  - Credit Mix: {len(credit_mix_features)} categories")
            if loan_type_features:
                print(f"  - Loan Types: {len(loan_type_features)} types")
        
        # Sample data
        print(f"\n{'='*80}")
        print("SAMPLE DATA (First 5 rows)")
        print("="*80)
        
        key_columns = ['loan_id', 'Customer_ID', 'application_date', 'loan_amt', 
                      'Age', 'Annual_Income', 'DTI', 'default_label']
        available_key_cols = [c for c in key_columns if c in df.columns]
        
        df.select(*available_key_cols).show(5, truncate=False)
        
        # Feature completeness
        print(f"\n{'='*80}")
        print("FEATURE COMPLETENESS (Top 10 with nulls)")
        print("="*80)
        
        null_counts = []
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            if null_count > 0:
                null_pct = (null_count / total_count) * 100
                null_counts.append((column, null_count, null_pct))
        
        null_counts.sort(key=lambda x: x[2], reverse=True)
        
        if null_counts:
            for col_name, count, pct in null_counts[:10]:
                print(f"  {col_name}: {pct:.1f}% ({count:,} nulls)")
        else:
            print("  All features are complete (no nulls)")
        
    except Exception as e:
        print(f"\n✗ Error analyzing gold output: {str(e)}")


def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process Silver → Gold layer for loan default prediction'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2023-01-01',
        help='Start date (YYYY-MM-DD) for batch processing'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-12-01',
        help='End date (YYYY-MM-DD) for batch processing'
    )
    parser.add_argument(
        '--single-date',
        type=str,
        default=None,
        help='Process single date only (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--silver-dir',
        type=str,
        default='/app/datamart/silver/',
        help='Silver layer base directory'
    )
    parser.add_argument(
        '--gold-dir',
        type=str,
        default='/app/datamart/gold/feature_store/',
        help='Gold layer feature store directory'
    )
    parser.add_argument(
        '--skip-verification',
        action='store_true',
        help='Skip silver layer verification'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*80)
    print("GOLD LAYER PROCESSING - CONFIGURATION")
    print("="*80)
    print(f"Silver directory: {args.silver_dir}")
    print(f"Gold directory: {args.gold_dir}")
    
    if args.single_date:
        dates_to_process = [args.single_date]
        print(f"Mode: Single date processing")
        print(f"Date: {args.single_date}")
    else:
        dates_to_process = generate_first_of_month_dates(args.start_date, args.end_date)
        print(f"Mode: Batch processing")
        print(f"Date range: {args.start_date} to {args.end_date}")
        print(f"Total dates: {len(dates_to_process)}")
    
    print("="*80)
    
    # Initialize Spark
    print("\nInitializing Spark...")
    spark = (
    pyspark.sql.SparkSession.builder
        .appName("dev")
        .master("local[*]")                   # keep local mode
        .config("spark.driver.memory", "6g")  # ↑ give the driver more heap (try 4g, 6g, 8g)
        .config("spark.driver.maxResultSize", "2g")  # protect against huge collects
        .config("spark.sql.shuffle.partitions", "16") # fewer shuffles for local runs
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # .config("spark.executor.memory", "6g")      # optional; mainly for cluster mode
        .getOrCreate()
)
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark initialized")
    
    # Verify silver tables exist
    if not args.skip_verification:
        status, all_available = verify_silver_tables(args.silver_dir)
        
        if not all_available:
            response = input("\nProceed anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    
    # Create gold directory
    if not os.path.exists(args.gold_dir):
        os.makedirs(args.gold_dir)
        print(f"\n✓ Created gold directory: {args.gold_dir}")
    
    # Process each date
    print("\n" + "="*80)
    print("PROCESSING GOLD LAYER")
    print("="*80)
    
    success_count = 0
    failed_dates = []
    
    for idx, date_str in enumerate(dates_to_process, 1):
        print(f"\n[{idx}/{len(dates_to_process)}] Processing {date_str}...")
        
        try:
            utils.data_processing_gold_table.process_gold_feature_store(
                date_str,
                args.silver_dir,
                args.gold_dir,
                spark
            )
            success_count += 1
            print(f"✓ Completed {date_str}")
            
        except Exception as e:
            print(f"✗ Failed {date_str}: {str(e)}")
            failed_dates.append((date_str, str(e)))
    
    # Summary
    print("\n" + "="*80)
    print("PROCESSING SUMMARY")
    print("="*80)
    print(f"Total dates processed: {len(dates_to_process)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_dates)}")
    
    if failed_dates:
        print("\nFailed dates:")
        for date_str, error in failed_dates:
            print(f"  - {date_str}: {error}")
    
    # Verify output
    verify_gold_output(args.gold_dir, spark)
    
    print("\n" + "="*80)
    print("GOLD PROCESSING COMPLETED")
    print("="*80)
    
    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()