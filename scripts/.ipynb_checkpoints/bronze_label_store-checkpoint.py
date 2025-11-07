import argparse
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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# To call this script: 
# python bronze_label_store.py --snapshotdate "2023-01-01"
# python bronze_label_store.py --snapshotdate "2023-01-01" --tables "lms_loan_daily,features_clickstream"

def main(snapshotdate, tables=None):
    """
    Process bronze tables for a specific snapshot date
    
    Args:
        snapshotdate: Date string in format 'YYYY-MM-DD'
        tables: Comma-separated list of table names to process, or None for all tables
    """
    print('\n\n---starting bronze layer job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("bronze_processing") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # Load arguments
    date_str = snapshotdate
    
    # Create bronze datalake directory
    bronze_directory = "/app/datamart/bronze/"
    
    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

    # Determine which tables to process
    if tables:
        # Process specific tables
        table_list = [t.strip() for t in tables.split(',')]
        print(f"Processing specific tables: {table_list}")
        
        for table_name in table_list:
            try:
                utils.data_processing_bronze_table.process_bronze_table(
                    date_str, 
                    bronze_directory, 
                    spark,
                    table_name
                )
            except Exception as e:
                print(f"Error processing {table_name}: {e}")
    else:
        # Process all tables
        print("Processing all bronze tables")
        utils.data_processing_bronze_table.process_all_bronze_tables(
            date_str, 
            bronze_directory, 
            spark
        )
    
    # End spark session
    spark.stop()
    
    print('\n\n---completed bronze layer job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process bronze layer tables for a specific snapshot date"
    )
    parser.add_argument(
        "--snapshotdate", 
        type=str, 
        required=True, 
        help="Snapshot date in format YYYY-MM-DD"
    )
    parser.add_argument(
        "--tables",
        type=str,
        required=False,
        default=None,
        help="Comma-separated list of tables to process (e.g., 'lms_loan_daily,features_clickstream'). If not provided, all tables will be processed."
    )
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.tables)