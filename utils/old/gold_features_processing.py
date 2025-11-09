"""
Gold Features Processing Utilities
Reusable functions for feature engineering (training and inference)
"""
import os
import glob
import logging

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, datediff, avg, stddev, sum as _sum, max as _max
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, DateType, IntegerType as IntType
from pyspark.sql.window import Window

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_silver_table(silver_directory, table_name, asof_date, spark):
    """Load silver table filtered by snapshot_date"""
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


def create_applications_dataframe_training(loan_df, asof_date):
    """
    Extract applications from loan_df for training
    Returns one row per loan_id with application details
    
    Args:
        loan_df: Loan daily DataFrame
        asof_date: As-of date for filtering
        
    Returns:
        DataFrame with columns: loan_id, Customer_ID, application_date, loan_amt, tenure
    """
    try:
        apps_df = loan_df.select(
            "loan_id",
            "Customer_ID",
            col("loan_start_date").alias("application_date"),
            "loan_amt",
            "tenure"
        ).distinct()
        
        apps_df = apps_df.filter(col("application_date") <= asof_date)
        
        logger.info(f"Created training applications: {apps_df.count()} unique loan_ids")
        return apps_df
        
    except Exception as e:
        logger.error(f"Error creating training applications: {str(e)}")
        raise


def create_applications_dataframe_inference(customer_id, application_date, spark, 
                                           loan_amt=10000, tenure=10):
    """
    Create application dataframe for inference (single application)
    
    Args:
        customer_id: Customer ID
        application_date: Application date (string 'YYYY-MM-DD' or date)
        spark: SparkSession
        loan_amt: Requested loan amount (default: 10000)
        tenure: Requested tenure in months (default: 10)
        
    Returns:
        DataFrame with one row containing application details
    """
    try:
        # Create single row DataFrame
        schema = StructType([
            StructField("Customer_ID", StringType(), False),
            StructField("application_date", DateType(), False),
            StructField("loan_amt", IntType(), False),
            StructField("tenure", IntType(), False)
        ])
        
        data = [(customer_id, application_date, loan_amt, tenure)]
        apps_df = spark.createDataFrame(data, schema)
        
        logger.info(f"Created inference application for customer: {customer_id}")
        return apps_df
        
    except Exception as e:
        logger.error(f"Error creating inference application: {str(e)}")
        raise


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def compute_capacity_features(apps_df, financials_df):
    """
    Compute capacity/affordability features from financials
    Single snapshot per customer
    """
    try:
        logger.info("Computing capacity features...")
        
        # Filter financials before application date
        financials_filtered = financials_df.join(
            apps_df.select("Customer_ID", "application_date"),
            on="Customer_ID",
            how="inner"
        ).filter(col("snapshot_date") < col("application_date"))
        
        # Select relevant columns
        financials_clean = financials_filtered.select(
            "Customer_ID", "Annual_Income", "Outstanding_Debt"
        ).distinct()
        
        # Join back to apps
        features_df = apps_df.join(financials_clean, on="Customer_ID", how="left")
        
        # DTI
        features_df = features_df.withColumn(
            "DTI",
            when(col("Annual_Income") > 0, col("Outstanding_Debt") / col("Annual_Income"))
            .otherwise(None)
        )
        
        # log_Annual_Income
        features_df = features_df.withColumn(
            "log_Annual_Income",
            when(col("Annual_Income") > 0, F.log(col("Annual_Income")))
            .otherwise(None)
        )
        
        
        logger.info(f"Capacity features: {features_df.count()} rows")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing capacity features: {str(e)}")
        raise


def compute_behavioral_features(features_df, financials_df):
    """
    Compute behavioral features from financials
    Single snapshot per customer
    """
    try:
        logger.info("Computing behavioral features...")
        
        # Filter financials before application date
        financials_filtered = financials_df.join(
            features_df.select("Customer_ID", "application_date"),
            on="Customer_ID",
            how="inner"
        ).filter(col("snapshot_date") < col("application_date"))
        
        # Select behavioral columns
        financials_clean = financials_filtered.select(
            "Customer_ID", "Payment_Behaviour", "Credit_Mix", "Type_of_Loan"
        ).distinct()
        
        # Join back
        features_df = features_df.join(financials_clean, on="Customer_ID", how="left")
        
        # One-hot encode Payment_Behaviour
        payment_behaviours = [
            "High_spent_Small_value_payments", 
            "Low_spent_Large_value_payments",
            "Low_spent_Medium_value_payments",
            "Low_spent_Small_value_payments",
            "High_spent_Medium_value_payments",
            "High_spent_Large_value_payments"
        ]
        
        for behaviour in payment_behaviours:
            col_name = f"Payment_Behaviour_{behaviour.replace(' ', '_').replace('-', '_')}"
            features_df = features_df.withColumn(
                col_name,
                when(col("Payment_Behaviour") == behaviour, 1).otherwise(0)
            )
        
        # One-hot encode Credit_Mix
        for credit_mix in ["Standard", "Good", "Bad"]:
            col_name = f"Credit_Mix_{credit_mix}"
            features_df = features_df.withColumn(
                col_name,
                when(col("Credit_Mix") == credit_mix, 1).otherwise(0)
            )
        
        # Multi-hot encode Type_of_Loan
        loan_types = [
            "Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan",
            "Mortgage Loan", "Student Loan", "Debt Consolidation Loan", "Payday Loan"
        ]
        
        for loan_type in loan_types:
            col_name = f"Type_of_Loan_{loan_type.replace(' ', '_').replace('-', '_')}"
            features_df = features_df.withColumn(
                col_name,
                when(col("Type_of_Loan").contains(loan_type), 1).otherwise(0)
            )
        
        # Drop original categorical columns
        features_df = features_df.drop("Payment_Behaviour", "Credit_Mix", "Type_of_Loan")
        
        logger.info(f"Behavioral features: {features_df.count()} rows")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing behavioral features: {str(e)}")
        raise


def compute_demographic_features(features_df, attributes_df):
    """
    Compute demographic features from attributes
    Single snapshot per customer
    """
    try:
        logger.info("Computing demographic features...")
        
        # Filter attributes before application date
        attributes_filtered = attributes_df.join(
            features_df.select("Customer_ID", "application_date"),
            on="Customer_ID",
            how="inner"
        ).filter(col("snapshot_date") < col("application_date"))
        
        # Select demographic columns
        attributes_clean = attributes_filtered.select(
            "Customer_ID", "Age", "Occupation"
        ).distinct()
        
        # Join back
        features_df = features_df.join(attributes_clean, on="Customer_ID", how="left")
        
        # Age band
        features_df = features_df.withColumn(
            "age_band",
            when(col("Age") < 25, "18-24")
            .when((col("Age") >= 25) & (col("Age") < 35), "25-34")
            .when((col("Age") >= 35) & (col("Age") < 45), "35-44")
            .when((col("Age") >= 45) & (col("Age") < 55), "45-54")
            .otherwise("55+")
        )
        
        logger.info(f"Demographic features: {features_df.count()} rows")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing demographic features: {str(e)}")
        raise


def compute_credit_depth_features(features_df, financials_df):
    """
    Compute credit depth features from financials
    Single snapshot per customer
    """
    try:
        logger.info("Computing credit depth features...")
        
        # Filter financials before application date
        financials_filtered = financials_df.join(
            features_df.select("Customer_ID", "application_date"),
            on="Customer_ID",
            how="inner"
        ).filter(col("snapshot_date") < col("application_date"))
        
        # Select credit depth columns
        financials_clean = financials_filtered.select(
            "Customer_ID", "Credit_History_Age", "Num_of_Loan"
        ).distinct()
        
        # Join back
        features_df = features_df.join(financials_clean, on="Customer_ID", how="left")
        
        # Parse Credit_History_Age
        features_df = features_df.withColumn(
            "Credit_History_Age_Year",
            F.regexp_extract("Credit_History_Age", r"(\d+) Years", 1).cast(IntegerType())
        )
        
        features_df = features_df.withColumn(
            "Credit_History_Age_Month",
            F.regexp_extract("Credit_History_Age", r"(\d+) Months", 1).cast(IntegerType())
        )
        
        # Rename and clean up
        features_df = features_df.withColumnRenamed("Num_of_Loan", "Num_of_Loan_active")
        features_df = features_df.drop("Credit_History_Age")
        
        logger.info(f"Credit depth features: {features_df.count()} rows")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing credit depth features: {str(e)}")
        raise


def compute_delinquency_features(features_df, financials_df):
    """
    Compute delinquency features from financials
    Single snapshot per customer
    """
    try:
        logger.info("Computing delinquency features...")
        
        # Filter financials before application date
        financials_filtered = financials_df.join(
            features_df.select("Customer_ID", "application_date"),
            on="Customer_ID",
            how="inner"
        ).filter(col("snapshot_date") < col("application_date"))
        
        # 3 months lookback
        financials_3m = financials_filtered.filter(
            datediff(col("application_date"), col("snapshot_date")).between(0, 90)
        )
        delayed_3m = financials_3m.groupBy("Customer_ID").agg(
            _sum("Num_of_Delayed_Payment").alias("Num_of_Delayed_Payment_3m")
        )
        
        # 6 months lookback
        financials_6m = financials_filtered.filter(
            datediff(col("application_date"), col("snapshot_date")).between(0, 180)
        )
        delayed_6m = financials_6m.groupBy("Customer_ID").agg(
            _sum("Num_of_Delayed_Payment").alias("Num_of_Delayed_Payment_6m")
        )
        
        # 12 months lookback
        financials_12m = financials_filtered.filter(
            datediff(col("application_date"), col("snapshot_date")).between(0, 365)
        )
        delayed_12m = financials_12m.groupBy("Customer_ID").agg(
            _sum("Num_of_Delayed_Payment").alias("Num_of_Delayed_Payment_12m"),
            _max("Num_of_Delayed_Payment").alias("max_dpd_prior")
        )
        
        # Join all delinquency features
        features_df = features_df.join(delayed_3m, on="Customer_ID", how="left")
        features_df = features_df.join(delayed_6m, on="Customer_ID", how="left")
        features_df = features_df.join(delayed_12m, on="Customer_ID", how="left")
        
        # Binary flag for 30+ DPD
        features_df = features_df.withColumn(
            "ever_30dpd_prior",
            when(col("max_dpd_prior") >= 30, 1).otherwise(0)
        )
        
        # Fill nulls
        features_df = features_df.fillna({
            "Num_of_Delayed_Payment_3m": 0,
            "Num_of_Delayed_Payment_6m": 0,
            "Num_of_Delayed_Payment_12m": 0,
            "max_dpd_prior": 0
        })
        
        logger.info(f"Delinquency features: {features_df.count()} rows")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing delinquency features: {str(e)}")
        raise


def compute_clickstream_features(features_df, clickstream_df):
    """
    Compute clickstream features
    Multiple snapshots per customer
    """
    try:
        logger.info("Computing clickstream features...")
        
        # Join clickstream with application info
        clickstream_filtered = clickstream_df.join(
            features_df.select("Customer_ID", "application_date"),
            on="Customer_ID",
            how="inner"
        ).filter(col("snapshot_date") < col("application_date"))
        
        # Get fe_ columns
        fe_columns = [c for c in clickstream_filtered.columns if c.startswith('fe_')]
        
        # 7-day window
        clickstream_7d = clickstream_filtered.filter(
            datediff(col("application_date"), col("snapshot_date")).between(0, 7)
        )
        
        agg_exprs_7d = []
        for fe_col in fe_columns:
            agg_exprs_7d.extend([
                _sum(fe_col).alias(f"{fe_col}_sum_7d"),
                avg(fe_col).alias(f"{fe_col}_mean_7d"),
                stddev(fe_col).alias(f"{fe_col}_std_7d")
            ])
        
        clickstream_7d_agg = clickstream_7d.groupBy("Customer_ID").agg(*agg_exprs_7d)
        features_df = features_df.join(clickstream_7d_agg, on="Customer_ID", how="left")
        
        # 30-day window
        clickstream_30d = clickstream_filtered.filter(
            datediff(col("application_date"), col("snapshot_date")).between(0, 30)
        )
        
        agg_exprs_30d = []
        for fe_col in fe_columns:
            agg_exprs_30d.extend([
                _sum(fe_col).alias(f"{fe_col}_sum_30d"),
                avg(fe_col).alias(f"{fe_col}_mean_30d"),
                stddev(fe_col).alias(f"{fe_col}_std_30d")
            ])
        
        clickstream_30d_agg = clickstream_30d.groupBy("Customer_ID").agg(*agg_exprs_30d)
        features_df = features_df.join(clickstream_30d_agg, on="Customer_ID", how="left")
        
        logger.info(f"Clickstream features: {features_df.count()} rows")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing clickstream features: {str(e)}")
        raise


def compute_application_features(features_df):
    """
    Compute application-specific features
    """
    try:
        logger.info("Computing application features...")
        
        monthly_rate = 0.12 / 12
        
        # Estimated EMI
        features_df = features_df.withColumn(
            "estimated_EMI",
            (col("loan_amt") * monthly_rate * F.pow(1 + monthly_rate, col("tenure"))) /
            (F.pow(1 + monthly_rate, col("tenure")) - 1)
        )
        
        # EMI to income ratio
        features_df = features_df.withColumn(
            "EMI_to_income",
            when(col("Annual_Income") > 0, 
                 col("estimated_EMI") / (col("Annual_Income") / 12))
            .otherwise(None)
        )
        
        # Requested amount/tenure
        features_df = features_df.withColumn("requested_amount", col("loan_amt"))
        features_df = features_df.withColumn("requested_tenure", col("tenure"))
        
        logger.info(f"Application features: {features_df.count()} rows")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing application features: {str(e)}")
        raise


def build_features(apps_df, silver_directory, asof_date, spark):
    """
    Build all features from historical tables
    Works for both training and inference
    
    Args:
        apps_df: Applications DataFrame (Customer_ID, application_date, loan_amt, tenure)
        silver_directory: Path to silver layer
        asof_date: As-of date for loading historical data
        spark: SparkSession
        
    Returns:
        DataFrame with all features
    """
    try:
        logger.info("Building features from historical data...")
        
        # Load historical tables
        financials_df = load_silver_table(silver_directory, "features_financials", asof_date, spark)
        attributes_df = load_silver_table(silver_directory, "features_attributes", asof_date, spark)
        clickstream_df = load_silver_table(silver_directory, "feature_clickstream", asof_date, spark)
        
        features_df = apps_df
        
        # Compute features
        if financials_df is not None:
            features_df = compute_capacity_features(features_df, financials_df)
            features_df = compute_behavioral_features(features_df, financials_df)
            features_df = compute_credit_depth_features(features_df, financials_df)
            features_df = compute_delinquency_features(features_df, financials_df)
        else:
            logger.warning("Financials data not available")
        
        if attributes_df is not None:
            features_df = compute_demographic_features(features_df, attributes_df)
        else:
            logger.warning("Attributes data not available")
        
        if clickstream_df is not None:
            features_df = compute_clickstream_features(features_df, clickstream_df)
        else:
            logger.warning("Clickstream data not available")
        
        features_df = compute_application_features(features_df)
        
        # Add snapshot_date
        features_df = features_df.withColumn("snapshot_date", col("application_date").cast("date"))
        
        logger.info(f"Features built: {features_df.count()} rows, {len(features_df.columns)} columns")
        return features_df
        
    except Exception as e:
        logger.error(f"Error building features: {str(e)}")
        raise