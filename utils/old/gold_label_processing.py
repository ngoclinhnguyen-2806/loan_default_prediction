import os
import glob
import logging

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.window import Window

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_silver_table(silver_directory, table_name, asof_date, spark):
    try:
        table_path = os.path.join(silver_directory, table_name, "*.parquet")
        
        if not glob.glob(os.path.join(silver_directory, table_name, "*.parquet")):
            logger.warning(f"No parquet files found for {table_name}")
            return None
        
        df = spark.read.parquet(table_path)
        df = df.filter(col("snapshot_date") <= asof_date)
        
        logger.info(f"Loaded {table_name} up to {asof_date}: {df.count()} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {table_name}: {str(e)}")
        return None


def create_applications_dataframe(loan_df, asof_date):
    try:
        window_spec = Window.partitionBy("loan_id").orderBy("snapshot_date")
        
        apps_df = loan_df.withColumn("row_num", F.row_number().over(window_spec)) \
                         .filter(col("row_num") == 1) \
                         .select(
                             "loan_id",
                             "Customer_ID",
                             col("loan_start_date").alias("application_date"),
                             "snapshot_date"
                         )
        
        apps_df = apps_df.filter(col("application_date") <= asof_date)
        
        logger.info(f"Created applications dataframe: {apps_df.count()} applications")
        return apps_df
        
    except Exception as e:
        logger.error(f"Error creating applications dataframe: {str(e)}")
        raise


def compute_labels(apps_df, loan_df, dpd_threshold=30, mob_threshold=0):
    try:
        logger.info(f"Computing labels (DPD>={dpd_threshold}, MOB>={mob_threshold})...")
        
        mature_loans = loan_df.filter(col("mob") >= mob_threshold)
        
        defaulted_loans = mature_loans.filter(col("dpd") >= dpd_threshold) \
                                     .select("loan_id") \
                                     .distinct() \
                                     .withColumn("default_label", lit(1))
        
        labels_df = apps_df.select("loan_id", "Customer_ID", "application_date").join(
            defaulted_loans,
            on="loan_id",
            how="left"
        ).fillna({"default_label": 0})
        
        # Deduplicate by loan_id: if any record has default_label=1, the loan is considered defaulted
        labels_df = labels_df.groupBy("loan_id", "Customer_ID", "application_date") \
                             .agg(F.sum("default_label").alias("default_sum")) \
                             .withColumn("default_label", 
                                       F.when(col("default_sum") > 0, 1).otherwise(0)) \
                             .select("loan_id", "Customer_ID", "application_date", "default_label")
        
        default_count = labels_df.filter(col("default_label") == 1).count()
        total_count = labels_df.count()
        default_rate = (default_count / total_count * 100) if total_count > 0 else 0
        
        logger.info(f"Labels computed: {default_count}/{total_count} defaults ({default_rate:.2f}%)")
        
        return labels_df
        
    except Exception as e:
        logger.error(f"Error computing labels: {str(e)}")
        raise


def process_gold_labels(asof_date_str, silver_directory, gold_label_directory, spark):
    try:
        logger.info("="*80)
        logger.info(f"Processing Gold Labels for {asof_date_str}")
        logger.info("="*80)
        
        if not os.path.exists(gold_label_directory):
            os.makedirs(gold_label_directory)
            logger.info(f"Created directory: {gold_label_directory}")
        
        loan_df = load_silver_table(silver_directory, "loan_daily", asof_date_str, spark)
        
        if loan_df is None:
            logger.error("Loan data not available, cannot proceed")
            return None
        
        apps_df = create_applications_dataframe(loan_df, asof_date_str)
        labels_df = compute_labels(apps_df, loan_df, dpd_threshold=30, mob_threshold=0)
        
        label_partition_name = f"gold_label_store_{asof_date_str.replace('-','_')}.parquet"
        label_filepath = os.path.join(gold_label_directory, label_partition_name)
        labels_df.write.mode("overwrite").parquet(label_filepath)
        logger.info(f"Saved labels to {label_filepath}")
        
        default_count = labels_df.filter(col("default_label") == 1).count()
        total_count = labels_df.count()
        default_rate = (default_count / total_count * 100) if total_count > 0 else 0
        
        logger.info("="*80)
        logger.info("Label Store Summary:")
        logger.info(f"  Total labels: {total_count}")
        logger.info(f"  Default rate: {default_rate:.2f}% ({default_count}/{total_count})")
        logger.info("="*80)
        
        return labels_df
        
    except Exception as e:
        logger.error(f"Error processing gold labels: {str(e)}")
        raise
