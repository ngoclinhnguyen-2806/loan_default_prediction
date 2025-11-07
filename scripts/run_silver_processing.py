"""
Silver Layer Processing Script
Run this to process Bronze → Silver independently

Usage:
    python run_silver_processing.py
    python run_silver_processing.py --start-date 2023-01-01 --end-date 2024-12-01
    python run_silver_processing.py --single-date 2023-06-01
    python run_silver_processing.py --table lms_loan_daily
"""

import os
import glob
import argparse
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

import utils.data_processing_silver_table


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


def verify_bronze_tables(bronze_directory, table_names=None):
    """
    Verify that required bronze tables exist
    
    Args:
        bronze_directory: Base bronze directory
        table_names: List of table names to check (default: all 4 tables)
    
    Returns:
        dict: Status of each table
    """
    if table_names is None:
        table_names = ['lms_loan_daily', 'feature_clickstream', 
                      'features_attributes', 'features_financials']
    
    print("\n" + "="*80)
    print("VERIFYING BRONZE LAYER TABLES")
    print("="*80)
    
    status = {}
    all_available = True
    
    for table_name in table_names:
        table_path = os.path.join(bronze_directory, table_name)
        
        if os.path.exists(table_path):
            csv_files = glob.glob(os.path.join(table_path, "*.csv"))
            if csv_files:
                status[table_name] = {
                    'exists': True,
                    'partition_count': len(csv_files),
                    'path': table_path
                }
                print(f"✓ {table_name}: {len(csv_files)} partitions found")
            else:
                status[table_name] = {
                    'exists': False,
                    'partition_count': 0,
                    'path': table_path
                }
                print(f"✗ {table_name}: Directory exists but no CSV files")
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
        print("\n⚠ WARNING: Some bronze tables are missing!")
        print("Run bronze layer processing first.")
    
    return status, all_available


def verify_silver_output(silver_directory, spark):
    """
    Verify and summarize silver layer output
    
    Args:
        silver_directory: Silver layer base directory
        spark: SparkSession
    """
    print("\n" + "="*80)
    print("SILVER LAYER OUTPUT SUMMARY")
    print("="*80)
    
    table_names = ['loan_daily', 'feature_clickstream', 
                  'features_attributes', 'features_financials']
    
    for table_name in table_names:
        table_path = os.path.join(silver_directory, table_name)
        
        if not os.path.exists(table_path):
            print(f"\n✗ {table_name}: Directory not found")
            continue
        
        parquet_files = glob.glob(os.path.join(table_path, "*.parquet"))
        
        if not parquet_files:
            print(f"\n✗ {table_name}: No parquet files found")
            continue
        
        print(f"\n✓ {table_name}:")
        print(f"  Total partitions: {len(parquet_files)}")
        
        # Try to load and analyze
        try:
            df = spark.read.option("mergeSchema", "true").parquet(*parquet_files)
            
            total_rows = df.count()
            unique_customers = df.select("Customer_ID").distinct().count()
            
            print(f"  Total rows: {total_rows:,}")
            print(f"  Unique customers: {unique_customers:,}")
            print(f"  Columns: {len(df.columns)}")
            
            # Date range
            if "snapshot_date" in df.columns:
                date_stats = df.agg(
                    F.min("snapshot_date").alias("min_date"),
                    F.max("snapshot_date").alias("max_date")
                ).collect()[0]
                print(f"  Date range: {date_stats['min_date']} to {date_stats['max_date']}")
            
            # Sample schema
            print(f"  Sample columns: {', '.join(df.columns[:5])}...")
            
        except Exception as e:
            print(f"  ⚠ Error reading parquet: {str(e)}")


def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Process Bronze → Silver layer'
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
        '--table',
        type=str,
        default=None,
        help='Process specific table only (lms_loan_daily, feature_clickstream, features_attributes, features_financials)'
    )
    parser.add_argument(
        '--bronze-dir',
        type=str,
        default='/app/datamart/bronze/',
        help='Bronze layer base directory'
    )
    parser.add_argument(
        '--silver-dir',
        type=str,
        default='/app/datamart/silver/',
        help='Silver layer base directory'
    )
    parser.add_argument(
        '--skip-verification',
        action='store_true',
        help='Skip bronze layer verification'
    )
    
    args = parser.parse_args()
    
    # Validate table name if provided
    valid_tables = ['lms_loan_daily', 'feature_clickstream', 
                   'features_attributes', 'features_financials']
    if args.table and args.table not in valid_tables:
        print(f"❌ Invalid table name: {args.table}")
        print(f"Valid options: {', '.join(valid_tables)}")
        return
    
    # Print configuration
    print("\n" + "="*80)
    print("SILVER LAYER PROCESSING - CONFIGURATION")
    print("="*80)
    print(f"Bronze directory: {args.bronze_dir}")
    print(f"Silver directory: {args.silver_dir}")
    
    if args.single_date:
        dates_to_process = [args.single_date]
        print(f"Mode: Single date processing")
        print(f"Date: {args.single_date}")
    else:
        dates_to_process = generate_first_of_month_dates(args.start_date, args.end_date)
        print(f"Mode: Batch processing")
        print(f"Date range: {args.start_date} to {args.end_date}")
        print(f"Total dates: {len(dates_to_process)}")
    
    if args.table:
        print(f"Table filter: {args.table} only")
    else:
        print(f"Tables: All 4 tables")
    
    print("="*80)
    
    # Initialize Spark
    print("\nInitializing Spark...")
    spark = pyspark.sql.SparkSession.builder \
        .appName("silver_processing") \
        .master("local[*]") \
        .config("spark.sql.parquet.mergeSchema", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark initialized")
    
    # Verify bronze tables exist
    if not args.skip_verification:
        tables_to_check = [args.table] if args.table else None
        status, all_available = verify_bronze_tables(args.bronze_dir, tables_to_check)
        
        if not all_available:
            response = input("\nProceed anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                return
    
    # Create silver directory
    if not os.path.exists(args.silver_dir):
        os.makedirs(args.silver_dir)
        print(f"\n✓ Created silver directory: {args.silver_dir}")
    
    # Process each date
    print("\n" + "="*80)
    print("PROCESSING SILVER LAYER")
    print("="*80)
    
    success_count = 0
    failed_dates = []
    
    for idx, date_str in enumerate(dates_to_process, 1):
        print(f"\n[{idx}/{len(dates_to_process)}] Processing {date_str}...")
        
        try:
            # Process single table or all tables
            results = utils.data_processing_silver_table.process_silver_table(
                date_str,
                args.bronze_dir,
                args.silver_dir,
                spark,
                table_name=args.table  # None means process all
            )
            
            # Check results
            if args.table:
                # Single table mode
                if results is not None:
                    success_count += 1
                    print(f"✓ Completed {args.table} for {date_str}")
                else:
                    print(f"⚠ {args.table} returned None for {date_str}")
                    failed_dates.append((date_str, "Returned None"))
            else:
                # All tables mode
                if results and any(df is not None for df in results.values()):
                    success_count += 1
                    successful_tables = [t for t, df in results.items() if df is not None]
                    failed_tables = [t for t, df in results.items() if df is None]
                    
                    if successful_tables:
                        print(f"✓ Completed for {date_str}: {', '.join(successful_tables)}")
                    if failed_tables:
                        print(f"⚠ Failed tables: {', '.join(failed_tables)}")
                else:
                    print(f"✗ All tables failed for {date_str}")
                    failed_dates.append((date_str, "All tables failed"))
            
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
    verify_silver_output(args.silver_dir, spark)
    
    # Additional statistics
    print("\n" + "="*80)
    print("SILVER LAYER STATISTICS")
    print("="*80)
    
    tables_to_check = [args.table] if args.table else ['lms_loan_daily', 'feature_clickstream',
                                                        'features_attributes', 'features_financials']
    
    for table_name in tables_to_check:
        # Map to silver table names
        silver_table = 'loan_daily' if table_name == 'lms_loan_daily' else table_name
        table_path = os.path.join(args.silver_dir, silver_table)
        
        if os.path.exists(table_path):
            parquet_files = glob.glob(os.path.join(table_path, "*.parquet"))
            print(f"\n{silver_table}:")
            print(f"  Partitions created: {len(parquet_files)}")
            
            # Calculate total size
            total_size = sum(
                sum(os.path.getsize(os.path.join(root, f)) 
                    for f in files)
                for root, dirs, files in os.walk(table_path)
            )
            print(f"  Total size: {total_size / (1024*1024):.2f} MB")
    
    print("\n" + "="*80)
    print("SILVER PROCESSING COMPLETED")
    print("="*80)
    
    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()