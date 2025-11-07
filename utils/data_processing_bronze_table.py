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
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_directory, spark, table_name):
    """
    Process a single bronze table for a given snapshot date
    
    Args:
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
        bronze_directory: Base directory for bronze layer
        spark: SparkSession object
        table_name: Name of the table to process (e.g., 'lms_loan_daily', 'features_clickstream')
    
    Returns:
        DataFrame: Processed Spark DataFrame
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Define source file mapping
    source_file_mapping = {
        'lms_loan_daily': 'data/lms_loan_daily.csv',
        'feature_clickstream': 'data/feature_clickstream.csv',
        'features_attributes': 'data/features_attributes.csv',
        'features_financials': 'data/features_financials.csv'
    }
    
    # Get the source file path
    csv_file_path = source_file_mapping.get(table_name)
    
    if csv_file_path is None:
        raise ValueError(f"Unknown table name: {table_name}. Valid options: {list(source_file_mapping.keys())}")
    
    if not os.path.exists(csv_file_path):
        print(f"Warning: Source file not found: {csv_file_path}")
        return None
    
    # Create table-specific directory
    table_directory = os.path.join(bronze_directory, table_name)
    if not os.path.exists(table_directory):
        os.makedirs(table_directory)
    
    # Load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)
    
    # Filter by snapshot_date if the column exists
    if 'snapshot_date' in df.columns:
        df = df.filter(col('snapshot_date') == snapshot_date)
    else:
        print(f"Warning: 'snapshot_date' column not found in {table_name}")
    
    print(f"{table_name} - {snapshot_date_str} row count: {df.count()}")
    
    # Save bronze table to datamart
    partition_name = f"bronze_{table_name}_{snapshot_date_str.replace('-','_')}.csv"
    filepath = os.path.join(table_directory, partition_name)
    df.toPandas().to_csv(filepath, index=False)
    print(f"Saved to: {filepath}")
    
    return df


def process_all_bronze_tables(snapshot_date_str, bronze_directory, spark):
    """
    Process all bronze tables for a given snapshot date
    
    Args:
        snapshot_date_str: Date string in format 'YYYY-MM-DD'
        bronze_directory: Base directory for bronze layer
        spark: SparkSession object
    
    Returns:
        dict: Dictionary of table_name -> DataFrame
    """
    table_names = [
        'lms_loan_daily',
        'feature_clickstream', 
        'features_attributes',
        'features_financials'
    ]
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Processing Bronze Tables for {snapshot_date_str}")
    print(f"{'='*60}\n")
    
    for table_name in table_names:
        print(f"Processing {table_name}...")
        try:
            df = process_bronze_table(snapshot_date_str, bronze_directory, spark, table_name)
            results[table_name] = df
            print(f"✓ {table_name} completed\n")
        except Exception as e:
            print(f"✗ Error processing {table_name}: {e}\n")
            results[table_name] = None
    
    print(f"{'='*60}")
    print(f"Bronze layer processing completed for {snapshot_date_str}")
    print(f"{'='*60}\n")
    
    return results