"""
Test Gold Layer Processing for Single Date
Quick script to test feature engineering on one snapshot date

Usage:
    python test_gold_processing.py
    python test_gold_processing.py --date 2023-06-01
"""

import os
import argparse
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col

import utils.data_processing_gold_table


def check_silver_tables(silver_directory, date_str, spark):
    """
    Check if silver tables exist and show their contents for the date
    
    Args:
        silver_directory: Base silver directory
        date_str: Date string to check
        spark: SparkSession
    
    Returns:
        dict: Information about each table
    """
    print("\n" + "="*80)
    print(f"CHECKING SILVER TABLES FOR {date_str}")
    print("="*80)
    
    tables = {
        'loan_daily': None,
        'feature_clickstream': None,
        'features_attributes': None,
        'features_financials': None
    }
    
    for table_name in tables.keys():
        table_path = os.path.join(silver_directory, table_name)
        
        if not os.path.exists(table_path):
            print(f"\n✗ {table_name}: Directory not found at {table_path}")
            continue
        
        try:
            # Try to load all partitions
            df = spark.read.parquet(os.path.join(table_path, "*.parquet"))
            
            # Filter to our date
            df_date = df.filter(col("snapshot_date") <= date_str)
            
            total_rows = df_date.count()
            unique_customers = df_date.select("Customer_ID").distinct().count()
            
            tables[table_name] = {
                'exists': True,
                'rows': total_rows,
                'customers': unique_customers,
                'columns': len(df_date.columns)
            }
            
            print(f"\n✓ {table_name}:")
            print(f"    Rows (up to {date_str}): {total_rows:,}")
            print(f"    Unique customers: {unique_customers:,}")
            print(f"    Columns: {len(df_date.columns)}")
            print(f"    Sample columns: {', '.join(df_date.columns[:5])}")
            
        except Exception as e:
            print(f"\n✗ {table_name}: Error loading - {str(e)}")
            tables[table_name] = {'exists': False, 'error': str(e)}
    
    print("="*80)
    
    # Check if we have minimum required tables
    has_loan = tables['loan_daily'] and tables['loan_daily'].get('exists', False)
    
    if not has_loan:
        print("\n⚠ WARNING: loan_daily table is required but not found!")
        return tables, False
    
    return tables, True


def test_feature_computation(date_str, silver_dir, gold_dir, spark):
    """
    Test feature store creation for a single date with detailed logging
    
    Args:
        date_str: Date string to process
        silver_dir: Silver directory path
        gold_dir: Gold directory path
        spark: SparkSession
    
    Returns:
        DataFrame or None
    """
    print("\n" + "="*80)
    print(f"TESTING GOLD FEATURE STORE CREATION FOR {date_str}")
    print("="*80)
    
    try:
        # Create gold directory if needed
        if not os.path.exists(gold_dir):
            os.makedirs(gold_dir)
            print(f"✓ Created gold directory: {gold_dir}")
        
        # Run the processing
        print(f"\nStarting feature engineering pipeline...")
        print("This will compute:")
        print("  - Capacity features (DTI, income)")
        print("  - Credit depth features")
        print("  - Delinquency history")
        print("  - Behavioral patterns")
        print("  - Demographics")
        print("  - Clickstream activity")
        print("  - Application features")
        print("  - Default labels")
        print()
        
        df = utils.data_processing_gold_table.process_gold_feature_store(
            date_str,
            silver_dir,
            gold_dir,
            spark
        )
        
        if df is None:
            print("\n✗ Processing returned None")
            return None
        
        print(f"\n✓ Feature store created successfully!")
        return df
        
    except Exception as e:
        print(f"\n✗ Error during processing: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        return None


def analyze_output(df, date_str):
    """
    Analyze the output feature store DataFrame
    
    Args:
        df: Output DataFrame from gold processing
        date_str: Date that was processed
    """
    if df is None:
        print("\n✗ No DataFrame to analyze")
        return
    
    print("\n" + "="*80)
    print("DETAILED FEATURE STORE ANALYSIS")
    print("="*80)
    
    # Basic stats
    total_rows = df.count()
    total_cols = len(df.columns)
    
    print(f"\nBasic Statistics:")
    print(f"  Total applications: {total_rows:,}")
    print(f"  Total features: {total_cols}")
    print(f"  As-of date: {date_str}")
    
    # Customer stats
    unique_customers = df.select("Customer_ID").distinct().count()
    print(f"  Unique customers: {unique_customers:,}")
    
    # Loan amount stats
    loan_stats = df.select(
        F.min("loan_amt").alias("min_loan"),
        F.max("loan_amt").alias("max_loan"),
        F.avg("loan_amt").alias("avg_loan")
    ).collect()[0]
    
    print(f"\nLoan Amount Distribution:")
    print(f"  Min: ${loan_stats['min_loan']:,.2f}")
    print(f"  Max: ${loan_stats['max_loan']:,.2f}")
    print(f"  Avg: ${loan_stats['avg_loan']:,.2f}")
    
    # Application date range
    if "application_date" in df.columns:
        date_range = df.agg(
            F.min("application_date").alias("min_date"),
            F.max("application_date").alias("max_date")
        ).collect()[0]
        print(f"\nApplication Date Range:")
        print(f"  From: {date_range['min_date']}")
        print(f"  To: {date_range['max_date']}")
    
    # Default rate
    if "default_label" in df.columns:
        default_count = df.filter(col("default_label") == 1).count()
        default_rate = (default_count / total_rows * 100) if total_rows > 0 else 0
        
        print(f"\nDefault Statistics:")
        print(f"  Defaults: {default_count:,} ({default_rate:.2f}%)")
        print(f"  Non-defaults: {total_rows - default_count:,} ({100 - default_rate:.2f}%)")
    
    # Feature completeness
    print(f"\n{'='*80}")
    print("FEATURE COMPLETENESS")
    print("="*80)
    
    null_stats = []
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        if null_count > 0:
            null_pct = (null_count / total_rows) * 100
            null_stats.append((column, null_count, null_pct))
    
    null_stats.sort(key=lambda x: x[2], reverse=True)
    
    if null_stats:
        print(f"\nFeatures with missing values (top 15):")
        for col_name, count, pct in null_stats[:15]:
            print(f"  {col_name:40s}: {pct:5.1f}% ({count:,} nulls)")
        
        complete_features = total_cols - len(null_stats)
        print(f"\nComplete features (no nulls): {complete_features}/{total_cols}")
    else:
        print(f"\n✓ All {total_cols} features are complete (no missing values)")
    
    # Feature categories count
    print(f"\n{'='*80}")
    print("FEATURE CATEGORIES")
    print("="*80)
    
    categories = {
        'Capacity': ['DTI', 'log_Annual_Income', 'income_band'],
        'Credit Depth': ['Credit_History_Age_Year', 'Num_of_Loan_active'],
        'Delinquency': ['Num_of_Delayed_Payment_3m', 'ever_30dpd_prior'],
        'Demographics': ['Age', 'age_band', 'Occupation'],
        'Behavioral': ['Payment_Behaviour', 'Credit_Mix', 'Type_of_Loan'],
        'Application': ['estimated_EMI', 'EMI_to_income'],
        'Clickstream': [c for c in df.columns if c.startswith('fe_')],
    }
    
    for category, features in categories.items():
        if category == 'Clickstream':
            available = features  # Already filtered
        else:
            available = [f for f in features if f in df.columns]
        
        if available:
            print(f"\n{category}: {len(available)} features")
            for f in available[:3]:
                print(f"  - {f}")
            if len(available) > 3:
                print(f"  ... and {len(available) - 3} more")
    
    # Sample key features
    print(f"\n{'='*80}")
    print("SAMPLE DATA - KEY FEATURES")
    print("="*80)
    
    key_features = ['loan_id', 'Customer_ID', 'application_date', 'loan_amt', 
                   'Age', 'Annual_Income', 'DTI', 'Num_of_Loan_active',
                   'Num_of_Delayed_Payment_12m', 'default_label']
    
    available_features = [f for f in key_features if f in df.columns]
    
    print(f"\nShowing first 10 rows:")
    df.select(*available_features).show(10, truncate=False)
    
    # Numeric feature summary
    print(f"\n{'='*80}")
    print("NUMERIC FEATURE SUMMARY")
    print("="*80)
    
    numeric_features = ['DTI', 'Age', 'Annual_Income', 'loan_amt', 
                       'Num_of_Loan_active', 'Num_of_Delayed_Payment_12m']
    available_numeric = [f for f in numeric_features if f in df.columns]
    
    if available_numeric:
        print(f"\nDescriptive statistics for key numeric features:")
        df.select(*available_numeric).describe().show()
    
    # Categorical distributions
    print(f"\n{'='*80}")
    print("CATEGORICAL DISTRIBUTIONS")
    print("="*80)
    
    if 'income_band' in df.columns:
        print(f"\nIncome Band Distribution:")
        df.groupBy('income_band').count().orderBy('count', ascending=False).show()
    
    if 'age_band' in df.columns:
        print(f"\nAge Band Distribution:")
        df.groupBy('age_band').count().orderBy('count', ascending=False).show()
    
    if 'Credit_Mix' in df.columns:
        print(f"\nCredit Mix Distribution:")
        df.groupBy('Credit_Mix').count().orderBy('count', ascending=False).show()


def main():
    """Main execution function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Test Gold layer processing for a single date'
    )
    parser.add_argument(
        '--date',
        type=str,
        default='2023-01-01',
        help='Snapshot date to process (YYYY-MM-DD)'
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
        default='/app/datamart/gold/test_feature_store/',
        help='Gold layer output directory (will create if not exists)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print("GOLD LAYER FEATURE ENGINEERING - TEST SCRIPT")
    print("="*80)
    print(f"Date to process: {args.date}")
    print(f"Silver directory: {args.silver_dir}")
    print(f"Gold directory: {args.gold_dir}")
    print("="*80)
    
    # Validate date format
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"\n✗ Invalid date format: {args.date}")
        print("Please use YYYY-MM-DD format (e.g., 2023-01-01)")
        return
    
    # Initialize Spark
    print("\nInitializing Spark...")
    spark = pyspark.sql.SparkSession.builder \
        .appName("test_gold_processing") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    print("✓ Spark session created")
    
    # Check silver tables
    tables_info, can_proceed = check_silver_tables(args.silver_dir, args.date, spark)
    
    if not can_proceed:
        print("\n✗ Cannot proceed - required silver tables missing")
        print("Run bronze→silver processing first")
        spark.stop()
        return
    
    # Confirm before proceeding
    print("\n" + "="*80)
    response = input("Proceed with gold layer processing? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        spark.stop()
        return
    
    # Run the test
    df = test_feature_computation(args.date, args.silver_dir, args.gold_dir, spark)
    
    if df is not None:
        # Analyze the results
        analyze_output(df, args.date)
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nOutput saved to: {args.gold_dir}")
        print(f"Partition file: gold_feature_store_{args.date.replace('-', '_')}.parquet")
    else:
        print("\n" + "="*80)
        print("TEST FAILED")
        print("="*80)
        print("\nCheck the error messages above for details")
    
    # Cleanup
    spark.stop()
    print("\n✓ Spark session stopped")


if __name__ == "__main__":
    main()