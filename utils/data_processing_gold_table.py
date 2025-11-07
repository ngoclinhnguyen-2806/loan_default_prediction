import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, datediff, months_between, avg, stddev, sum as _sum, count, max as _max, min as _min
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType
from pyspark.sql.window import Window

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_silver_table(silver_directory, table_name, asof_date, spark):
    """
    Load all partitions of a silver table up to and including asof_date
    
    Args:
        silver_directory: Base silver directory
        table_name: Name of the table
        asof_date: As-of date (string 'YYYY-MM-DD' or date object)
        spark: SparkSession
    
    Returns:
        DataFrame filtered to snapshot_date <= asof_date
    """
    try:
        table_path = os.path.join(silver_directory, table_name, "*.parquet")
        
        if not glob.glob(os.path.join(silver_directory, table_name, "*.parquet")):
            logger.warning(f"No parquet files found for {table_name}")
            return None
        
        df = spark.read.parquet(table_path)
        
        # Filter to historical data only (no future peeking)
        df = df.filter(col("snapshot_date") <= asof_date)
        
        logger.info(f"Loaded {table_name} up to {asof_date}: {df.count()} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {table_name}: {str(e)}")
        return None


def create_applications_dataframe(loan_df, asof_date):
    """
    Extract application-time records from loan data
    Uses the earliest record for each loan_id as the application time
    
    Args:
        loan_df: Silver loan_daily DataFrame
        asof_date: As-of date for feature computation
    
    Returns:
        DataFrame with (loan_id, Customer_ID, application_date, loan_amt, tenure)
    """
    try:
        # Get the first record for each loan (application time)
        window_spec = Window.partitionBy("loan_id").orderBy("snapshot_date")
        
        apps_df = loan_df.withColumn("row_num", F.row_number().over(window_spec)) \
                         .filter(col("row_num") == 1) \
                         .select(
                             "loan_id",
                             "Customer_ID",
                             col("loan_start_date").alias("application_date"),
                             "loan_amt",
                             "tenure",
                             "snapshot_date"
                         )
        
        # Only keep applications up to asof_date
        apps_df = apps_df.filter(col("application_date") <= asof_date)
        
        logger.info(f"Created applications dataframe: {apps_df.count()} applications")
        return apps_df
        
    except Exception as e:
        logger.error(f"Error creating applications dataframe: {str(e)}")
        raise


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def compute_capacity_features(apps_df, financials_df):
    """
    Compute capacity/affordability features
    - DTI (Debt to Income)
    - log_Annual_Income
    - income_band
    
    Args:
        apps_df: Applications DataFrame
        financials_df: Financial features DataFrame (filtered to snapshot_date <= app_date)
    
    Returns:
        DataFrame with capacity features
    """
    try:
        logger.info("Computing capacity/affordability features...")
        
        # Get latest financial snapshot for each customer before application
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        
        latest_financials = financials_df.withColumn("row_num", F.row_number().over(window_spec)) \
                                         .filter(col("row_num") == 1) \
                                         .select("Customer_ID", "Annual_Income", "Outstanding_Debt", 
                                                "Num_of_Loan", "snapshot_date")
        
        # Join with applications
        features_df = apps_df.join(latest_financials, on="Customer_ID", how="left")
        
        # DTI = Outstanding_Debt / Annual_Income
        features_df = features_df.withColumn(
            "DTI",
            when(col("Annual_Income") > 0, col("Outstanding_Debt") / col("Annual_Income"))
            .otherwise(None)
        )
        
        # log_Annual_Income (handle zeros/nulls)
        features_df = features_df.withColumn(
            "log_Annual_Income",
            when(col("Annual_Income") > 0, F.log(col("Annual_Income")))
            .otherwise(None)
        )
        
        # income_band
        features_df = features_df.withColumn(
            "income_band",
            when(col("Annual_Income") < 20000, "0-20k")
            .when((col("Annual_Income") >= 20000) & (col("Annual_Income") < 50000), "20-50k")
            .when((col("Annual_Income") >= 50000) & (col("Annual_Income") < 100000), "50-100k")
            .otherwise("100k+")
        )
        
        logger.info("Capacity features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing capacity features: {str(e)}")
        raise
        

def compute_behavioral_features(features_df, financials_df):
    """
    Compute behavioral/categorical features
    - Payment_Behaviour (one-hot encoded)
    - Credit_Mix (one-hot encoded)
    - Type_of_Loan multi-hot flags
    
    Args:
        features_df: Features DataFrame
        financials_df: Financial features DataFrame
    
    Returns:
        DataFrame with behavioral features
    """
    try:
        logger.info("Computing behavioral features...")
        
        # Get latest financial record
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        latest_financials = financials_df.withColumn("row_num", F.row_number().over(window_spec)) \
                                         .filter(col("row_num") == 1) \
                                         .select("Customer_ID", "Payment_Behaviour", 
                                                "Credit_Mix", "Type_of_Loan")
        
        # Join with features
        features_df = features_df.join(
            latest_financials.select("Customer_ID", "Payment_Behaviour", 
                                    "Credit_Mix", "Type_of_Loan"),
            on="Customer_ID",
            how="left"
        )
        
        # One-hot encode Payment_Behaviour
        payment_behaviours = ["High_spent_Small_value_payments", 
                             "Low_spent_Large_value_payments",
                             "Low_spent_Medium_value_payments",
                             "Low_spent_Small_value_payments",
                             "High_spent_Medium_value_payments",
                             "High_spent_Large_value_payments"]
        
        for behaviour in payment_behaviours:
            col_name = f"Payment_Behaviour_{behaviour.replace(' ', '_').replace('-', '_')}"
            features_df = features_df.withColumn(
                col_name,
                when(col("Payment_Behaviour") == behaviour, 1).otherwise(0)
            )
        
        # One-hot encode Credit_Mix
        credit_mixes = ["Standard", "Good", "Bad"]
        for mix in credit_mixes:
            features_df = features_df.withColumn(
                f"Credit_Mix_{mix}",
                when(col("Credit_Mix") == mix, 1).otherwise(0)
            )
        
        # Multi-hot encode Type_of_Loan (assuming comma-separated values)
        loan_types = ["Auto Loan", "Credit-Builder Loan", "Personal Loan", 
                     "Home Equity Loan", "Mortgage Loan", "Student Loan",
                     "Debt Consolidation Loan", "Payday Loan"]
        
        for loan_type in loan_types:
            col_name = f"Type_of_Loan_{loan_type.replace(' ', '_').replace('-', '_')}"
            features_df = features_df.withColumn(
                col_name,
                when(col("Type_of_Loan").contains(loan_type), 1).otherwise(0)
            )
        
        # Count number of loan types
        features_df = features_df.withColumn(
            "loan_type_count",
            F.size(F.split(col("Type_of_Loan"), ","))
        )
        
        logger.info("Behavioral features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing behavioral features: {str(e)}")
        raise


def compute_demographic_features(features_df, attributes_df):
    """
    Compute demographic features from attributes
    - age_clean (raw and binned)
    - Occupation (one-hot)
    
    Args:
        features_df: Features DataFrame
        attributes_df: Attributes DataFrame
    
    Returns:
        DataFrame with demographic features
    """
    try:
        logger.info("Computing demographic features...")
        
        # Get latest attributes
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        latest_attrs = attributes_df.withColumn("row_num", F.row_number().over(window_spec)) \
                                   .filter(col("row_num") == 1) \
                                   .select("Customer_ID", "age_clean", "Occupation")
        
        # Join with features
        features_df = features_df.join(
            latest_attrs,
            on="Customer_ID",
            how="left"
        )
        
        # Age binning
        features_df = features_df.withColumn(
            "age_band",
            when(col("age_clean") < 25, "18-24")
            .when((col("age_clean") >= 25) & (col("age_clean") < 35), "25-34")
            .when((col("age_clean") >= 35) & (col("age_clean") < 45), "35-44")
            .when((col("age_clean") >= 45) & (col("age_clean") < 55), "45-54")
            .otherwise("55+")
        )
        
        # One-hot encode Occupation (top categories)
        occupations = ["Scientist", "Teacher", "Engineer", "Entrepreneur", 
                      "Developer", "Lawyer", "Media_Manager", "Doctor",
                      "Journalist", "Manager", "Accountant", "Musician",
                      "Mechanic", "Writer", "Architect"]
        
        for occ in occupations:
            features_df = features_df.withColumn(
                f"Occupation_{occ}",
                when(col("Occupation") == occ, 1).otherwise(0)
            )
        
        logger.info("Demographic features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing demographic features: {str(e)}")
        raise


def compute_clickstream_features(features_df, clickstream_df):
    """
    Compute clickstream behavioral features
    - Aggregates over 7-day and 30-day windows
    - Sum, mean, variance of fe_1 to fe_20
    
    Args:
        features_df: Features DataFrame
        clickstream_df: Clickstream features DataFrame
    
    Returns:
        DataFrame with clickstream features
    """
    try:
        logger.info("Computing clickstream features...")
        
        # 7-day window aggregates
        clickstream_7d = features_df.alias("f").join(
            clickstream_df.alias("cs"),
            on="Customer_ID",
            how="left"
        ).filter(
            datediff(col("f.application_date"), col("cs.snapshot_date")).between(0, 7)
        )
        
        # Compute aggregates for fe_1 to fe_20
        agg_exprs_7d = []
        for i in range(1, 21):
            fe_col = f"fe_{i}"
            agg_exprs_7d.extend([
                _sum(fe_col).alias(f"{fe_col}_sum_7d"),
                avg(fe_col).alias(f"{fe_col}_mean_7d")
            ])
        
        clickstream_7d_agg = clickstream_7d.groupBy("f.loan_id").agg(*agg_exprs_7d)
        
        features_df = features_df.join(clickstream_7d_agg, on="loan_id", how="left")
        
        # 30-day window aggregates
        clickstream_30d = features_df.alias("f").join(
            clickstream_df.alias("cs"),
            on="Customer_ID",
            how="left"
        ).filter(
            datediff(col("f.application_date"), col("cs.snapshot_date")).between(0, 30)
        )
        
        agg_exprs_30d = []
        for i in range(1, 21):
            fe_col = f"fe_{i}"
            agg_exprs_30d.extend([
                _sum(fe_col).alias(f"{fe_col}_sum_30d"),
                avg(fe_col).alias(f"{fe_col}_mean_30d"),
                stddev(fe_col).alias(f"{fe_col}_std_30d")
            ])
        
        clickstream_30d_agg = clickstream_30d.groupBy("f.loan_id").agg(*agg_exprs_30d)
        
        features_df = features_df.join(clickstream_30d_agg, on="loan_id", how="left")
        
        logger.info("Clickstream features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing clickstream features: {str(e)}")
        raise


def compute_application_features(features_df):
    """
    Compute application-specific features
    - EMI to income ratio (estimated)
    
    Args:
        features_df: Features DataFrame with loan_amt, tenure, Annual_Income
    
    Returns:
        DataFrame with application features
    """
    try:
        logger.info("Computing application features...")
        
        # Estimate EMI using simple interest approximation
        # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        # For simplicity, assume 12% annual rate
        monthly_rate = 0.12 / 12
        
        features_df = features_df.withColumn(
            "estimated_EMI",
            (col("loan_amt") * monthly_rate * F.pow(1 + monthly_rate, col("tenure"))) /
            (F.pow(1 + monthly_rate, col("tenure")) - 1)
        )
        
        # EMI to monthly income ratio
        features_df = features_df.withColumn(
            "EMI_to_income",
            when(col("Annual_Income") > 0, 
                 col("estimated_EMI") / (col("Annual_Income") / 12))
            .otherwise(None)
        )
        
        # Requested amount features
        features_df = features_df.withColumn("requested_amount", col("loan_amt"))
        features_df = features_df.withColumn("requested_tenure", col("tenure"))
        
        logger.info("Application features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing application features: {str(e)}")
        raise


# =============================================================================
# LABEL COMPUTATION
# =============================================================================

def compute_labels(apps_df, loan_df, dpd_threshold=30, mob_threshold=0):
    """
    Compute default labels for applications
    Label = 1 if loan ever reaches dpd >= dpd_threshold after mob >= mob_threshold
    
    Args:
        apps_df: Applications DataFrame
        loan_df: Full loan history DataFrame
        dpd_threshold: DPD threshold for default (default: 30)
        mob_threshold: MOB threshold to consider (default: 0)
    
    Returns:
        DataFrame with (loan_id, default_label)
    """
    try:
        logger.info(f"Computing labels (DPD>={dpd_threshold}, MOB>={mob_threshold})...")
        
        # Filter to mature loans (mob >= threshold)
        mature_loans = loan_df.filter(col("mob") >= mob_threshold)
        
        # Check if ever defaulted
        defaulted_loans = mature_loans.filter(col("dpd") >= dpd_threshold) \
                                     .select("loan_id") \
                                     .distinct() \
                                     .withColumn("default_label", lit(1))
        
        # Join with all applications
        labels_df = apps_df.select("loan_id").join(
            defaulted_loans,
            on="loan_id",
            how="left"
        ).fillna({"default_label": 0})
        
        default_count = labels_df.filter(col("default_label") == 1).count()
        total_count = labels_df.count()
        default_rate = (default_count / total_count * 100) if total_count > 0 else 0
        
        logger.info(f"Labels computed: {default_count}/{total_count} defaults ({default_rate:.2f}%)")
        
        return labels_df
        
    except Exception as e:
        logger.error(f"Error computing labels: {str(e)}")
        raise


# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_gold_feature_store(asof_date_str, silver_directory, gold_directory, spark):
    """
    Process Gold layer feature store for loan default prediction
    Creates one row per application with all features computed as-of application time
    
    Args:
        asof_date_str: As-of date string 'YYYY-MM-DD'
        silver_directory: Silver layer base directory
        gold_directory: Gold layer base directory
        spark: SparkSession
    
    Returns:
        DataFrame with features and labels
    """
    try:
        logger.info("="*80)
        logger.info(f"Processing Gold Feature Store for {asof_date_str}")
        logger.info("="*80)
        
        # Create gold directory
        if not os.path.exists(gold_directory):
            os.makedirs(gold_directory)
            logger.info(f"Created directory: {gold_directory}")
        
        # Load all silver tables (up to asof_date)
        loan_df = load_silver_table(silver_directory, "loan_daily", asof_date_str, spark)
        financials_df = load_silver_table(silver_directory, "features_financials", asof_date_str, spark)
        attributes_df = load_silver_table(silver_directory, "features_attributes", asof_date_str, spark)
        clickstream_df = load_silver_table(silver_directory, "feature_clickstream", asof_date_str, spark)
        
        if loan_df is None:
            logger.error("Loan data not available, cannot proceed")
            return None
        
        # Create applications dataframe
        apps_df = create_applications_dataframe(loan_df, asof_date_str)
        
        # Start with applications as base
        features_df = apps_df
        
        # Compute feature families
        if financials_df is not None:
            features_df = compute_capacity_features(features_df, financials_df)
            features_df = compute_behavioral_features(features_df, financials_df)
        else:
            logger.warning("Financials data not available, skipping financial features")
        
        if attributes_df is not None:
            features_df = compute_demographic_features(features_df, attributes_df)
        else:
            logger.warning("Attributes data not available, skipping demographic features")
        
        if clickstream_df is not None:
            features_df = compute_clickstream_features(features_df, clickstream_df)
        else:
            logger.warning("Clickstream data not available, skipping clickstream features")
        
        # Application-specific features
        features_df = compute_application_features(features_df)
        
        # Compute labels
        labels_df = compute_labels(apps_df, loan_df, dpd_threshold=30, mob_threshold=0)
        
        # Join labels
        features_df = features_df.join(labels_df, on="loan_id", how="left")

        if "snapshot_date" in features_df.columns:
            features_df = features_df.drop("snapshot_date") # drop existing to avoid collision
        features_df = features_df.withColumn("snapshot_date", F.col("application_date").cast("date"))
        
        # Save to gold layer
        partition_name = f"gold_feature_store_{asof_date_str.replace('-','_')}.parquet"
        filepath = os.path.join(gold_directory, partition_name)

        # We want snapshot_date to equal application_date (as-of)
        features_df.write.mode("overwrite").parquet(filepath)
        logger.info(f"Saved feature store to {filepath}")
        
        # Summary statistics
        logger.info("="*80)
        logger.info("Feature Store Summary:")
        logger.info(f"  Total applications: {features_df.count()}")
        logger.info(f"  Total features: {len(features_df.columns)}")
        logger.info(f"  Date range: {asof_date_str}")
        logger.info("="*80)
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error processing gold feature store: {str(e)}")
        raise