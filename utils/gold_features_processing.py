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
                             "loan_amt",
                             "tenure",
                             "snapshot_date"
                         )
        
        apps_df = apps_df.filter(col("application_date") <= asof_date)
        
        logger.info(f"Created applications dataframe: {apps_df.count()} applications")
        return apps_df
        
    except Exception as e:
        logger.error(f"Error creating applications dataframe: {str(e)}")
        raise


def compute_capacity_features(apps_df, financials_df):
    try:
        logger.info("Computing capacity/affordability features...")
        
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        
        latest_financials = financials_df.withColumn("row_num", F.row_number().over(window_spec)) \
                                         .filter(col("row_num") == 1) \
                                         .select("Customer_ID", "Annual_Income", "Outstanding_Debt", 
                                                "Num_of_Loan", "snapshot_date")
        
        features_df = apps_df.join(latest_financials, on="Customer_ID", how="left")
        
        features_df = features_df.withColumn(
            "DTI",
            when(col("Annual_Income") > 0, col("Outstanding_Debt") / col("Annual_Income"))
            .otherwise(None)
        )
        
        features_df = features_df.withColumn(
            "log_Annual_Income",
            when(col("Annual_Income") > 0, F.log(col("Annual_Income")))
            .otherwise(None)
        )
        
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
    try:
        logger.info("Computing behavioral features...")
        
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        latest_financials = financials_df.withColumn("row_num", F.row_number().over(window_spec)) \
                                         .filter(col("row_num") == 1) \
                                         .select("Customer_ID", "Payment_Behaviour", 
                                                "Credit_Mix", "Type_of_Loan")
        
        features_df = features_df.join(
            latest_financials.select("Customer_ID", "Payment_Behaviour", 
                                    "Credit_Mix", "Type_of_Loan"),
            on="Customer_ID",
            how="left"
        )
        
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
        
        credit_mixes = ["Standard", "Good", "Bad"]
        for credit_mix in credit_mixes:
            col_name = f"Credit_Mix_{credit_mix}"
            features_df = features_df.withColumn(
                col_name,
                when(col("Credit_Mix") == credit_mix, 1).otherwise(0)
            )
        
        loan_types = ["Auto Loan", "Credit-Builder Loan", "Personal Loan", "Home Equity Loan",
                     "Mortgage Loan", "Student Loan", "Debt Consolidation Loan", "Payday Loan"]
        
        for loan_type in loan_types:
            col_name = f"Type_of_Loan_{loan_type.replace(' ', '_').replace('-', '_')}"
            features_df = features_df.withColumn(
                col_name,
                when(col("Type_of_Loan").contains(loan_type), 1).otherwise(0)
            )
        
        features_df = features_df.drop("Payment_Behaviour", "Credit_Mix", "Type_of_Loan")
        
        logger.info("Behavioral features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing behavioral features: {str(e)}")
        raise


def compute_demographic_features(features_df, attributes_df):
    try:
        logger.info("Computing demographic features...")
        
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        latest_attributes = attributes_df.withColumn("row_num", F.row_number().over(window_spec)) \
                                        .filter(col("row_num") == 1) \
                                        .select("Customer_ID", "Age", "Occupation")
        
        features_df = features_df.join(
            latest_attributes.select("Customer_ID", "Age", "Occupation"),
            on="Customer_ID",
            how="left"
        )
        
        features_df = features_df.withColumn(
            "age_band",
            when(col("Age") < 25, "18-24")
            .when((col("Age") >= 25) & (col("Age") < 35), "25-34")
            .when((col("Age") >= 35) & (col("Age") < 45), "35-44")
            .when((col("Age") >= 45) & (col("Age") < 55), "45-54")
            .otherwise("55+")
        )
        
        logger.info("Demographic features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing demographic features: {str(e)}")
        raise


def compute_credit_depth_features(features_df, financials_df):
    try:
        logger.info("Computing credit depth features...")
        
        window_spec = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
        latest_financials = financials_df.withColumn("row_num", F.row_number().over(window_spec)) \
                                         .filter(col("row_num") == 1) \
                                         .select("Customer_ID", "Credit_History_Age", "Num_of_Loan")
        
        features_df = features_df.join(
            latest_financials.select("Customer_ID", "Credit_History_Age", "Num_of_Loan"),
            on="Customer_ID",
            how="left"
        )
        
        features_df = features_df.withColumn(
            "Credit_History_Age_Year",
            F.regexp_extract("Credit_History_Age", r"(\d+) Years", 1).cast(IntegerType())
        )
        
        features_df = features_df.withColumn(
            "Credit_History_Age_Month",
            F.regexp_extract("Credit_History_Age", r"(\d+) Months", 1).cast(IntegerType())
        )
        
        features_df = features_df.withColumnRenamed("Num_of_Loan", "Num_of_Loan_active")
        features_df = features_df.drop("Credit_History_Age")
        
        logger.info("Credit depth features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing credit depth features: {str(e)}")
        raise


def compute_delinquency_features(features_df, financials_df):
    try:
        logger.info("Computing delinquency features...")
        
        financials_with_app = financials_df.join(
            features_df.select("Customer_ID", "application_date"),
            on="Customer_ID",
            how="inner"
        )
        
        financials_3m = financials_with_app.filter(
            datediff(col("application_date"), col("snapshot_date")).between(0, 90)
        )
        
        delayed_3m = financials_3m.groupBy("Customer_ID").agg(
            _sum("Num_of_Delayed_Payment").alias("Num_of_Delayed_Payment_3m")
        )
        
        financials_6m = financials_with_app.filter(
            datediff(col("application_date"), col("snapshot_date")).between(0, 180)
        )
        
        delayed_6m = financials_6m.groupBy("Customer_ID").agg(
            _sum("Num_of_Delayed_Payment").alias("Num_of_Delayed_Payment_6m")
        )
        
        financials_12m = financials_with_app.filter(
            datediff(col("application_date"), col("snapshot_date")).between(0, 365)
        )
        
        delayed_12m = financials_12m.groupBy("Customer_ID").agg(
            _sum("Num_of_Delayed_Payment").alias("Num_of_Delayed_Payment_12m"),
            _max("Num_of_Delayed_Payment").alias("max_dpd_prior")
        )
        
        features_df = features_df.join(delayed_3m, on="Customer_ID", how="left")
        features_df = features_df.join(delayed_6m, on="Customer_ID", how="left")
        features_df = features_df.join(delayed_12m, on="Customer_ID", how="left")
        
        features_df = features_df.withColumn(
            "ever_30dpd_prior",
            when(col("max_dpd_prior") >= 30, 1).otherwise(0)
        )
        
        features_df = features_df.fillna({
            "Num_of_Delayed_Payment_3m": 0,
            "Num_of_Delayed_Payment_6m": 0,
            "Num_of_Delayed_Payment_12m": 0,
            "max_dpd_prior": 0
        })
        
        logger.info("Delinquency features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing delinquency features: {str(e)}")
        raise


def compute_clickstream_features(features_df, clickstream_df):
    try:
        logger.info("Computing clickstream features...")
        
        clickstream_with_app = clickstream_df.join(
            features_df.select("loan_id", "application_date"),
            on="loan_id",
            how="inner"
        )
        
        clickstream_7d = clickstream_with_app.filter(
            datediff(col("application_date"), col("event_timestamp")).between(0, 7)
        )
        
        fe_columns = [c for c in clickstream_7d.columns if c.startswith('fe_')]
        
        agg_exprs_7d = []
        for fe_col in fe_columns:
            agg_exprs_7d.extend([
                _sum(fe_col).alias(f"{fe_col}_sum_7d"),
                avg(fe_col).alias(f"{fe_col}_mean_7d"),
                stddev(fe_col).alias(f"{fe_col}_std_7d")
            ])
        
        clickstream_7d_agg = clickstream_7d.groupBy("loan_id").agg(*agg_exprs_7d)
        features_df = features_df.join(clickstream_7d_agg, on="loan_id", how="left")
        
        clickstream_30d = clickstream_with_app.filter(
            datediff(col("application_date"), col("event_timestamp")).between(0, 30)
        )
        
        agg_exprs_30d = []
        for fe_col in fe_columns:
            agg_exprs_30d.extend([
                _sum(fe_col).alias(f"{fe_col}_sum_30d"),
                avg(fe_col).alias(f"{fe_col}_mean_30d"),
                stddev(fe_col).alias(f"{fe_col}_std_30d")
            ])
        
        clickstream_30d_agg = clickstream_30d.groupBy("loan_id").agg(*agg_exprs_30d)
        features_df = features_df.join(clickstream_30d_agg, on="loan_id", how="left")
        
        logger.info("Clickstream features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing clickstream features: {str(e)}")
        raise


def compute_application_features(features_df):
    try:
        logger.info("Computing application features...")
        
        monthly_rate = 0.12 / 12
        
        features_df = features_df.withColumn(
            "estimated_EMI",
            (col("loan_amt") * monthly_rate * F.pow(1 + monthly_rate, col("tenure"))) /
            (F.pow(1 + monthly_rate, col("tenure")) - 1)
        )
        
        features_df = features_df.withColumn(
            "EMI_to_income",
            when(col("Annual_Income") > 0, 
                 col("estimated_EMI") / (col("Annual_Income") / 12))
            .otherwise(None)
        )
        
        features_df = features_df.withColumn("requested_amount", col("loan_amt"))
        features_df = features_df.withColumn("requested_tenure", col("tenure"))
        
        logger.info("Application features computed successfully")
        return features_df
        
    except Exception as e:
        logger.error(f"Error computing application features: {str(e)}")
        raise


def process_gold_features(asof_date_str, silver_directory, gold_feature_directory, spark):
    try:
        logger.info("="*80)
        logger.info(f"Processing Gold Features for {asof_date_str}")
        logger.info("="*80)
        
        if not os.path.exists(gold_feature_directory):
            os.makedirs(gold_feature_directory)
            logger.info(f"Created directory: {gold_feature_directory}")
        
        loan_df = load_silver_table(silver_directory, "loan_daily", asof_date_str, spark)
        financials_df = load_silver_table(silver_directory, "features_financials", asof_date_str, spark)
        attributes_df = load_silver_table(silver_directory, "features_attributes", asof_date_str, spark)
        clickstream_df = load_silver_table(silver_directory, "feature_clickstream", asof_date_str, spark)
        
        if loan_df is None:
            logger.error("Loan data not available, cannot proceed")
            return None
        
        apps_df = create_applications_dataframe(loan_df, asof_date_str)
        features_df = apps_df
        
        if financials_df is not None:
            features_df = compute_capacity_features(features_df, financials_df)
            features_df = compute_behavioral_features(features_df, financials_df)
            features_df = compute_credit_depth_features(features_df, financials_df)
            features_df = compute_delinquency_features(features_df, financials_df)
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
        
        features_df = compute_application_features(features_df)
        
        if "snapshot_date" in features_df.columns:
            features_df = features_df.drop("snapshot_date")
        features_df = features_df.withColumn("snapshot_date", F.col("application_date").cast("date"))
        
        feature_partition_name = f"gold_feature_store_{asof_date_str.replace('-','_')}.parquet"
        feature_filepath = os.path.join(gold_feature_directory, feature_partition_name)
        features_df.write.mode("overwrite").parquet(feature_filepath)
        logger.info(f"Saved features to {feature_filepath}")
        
        logger.info("="*80)
        logger.info("Feature Store Summary:")
        logger.info(f"  Total applications: {features_df.count()}")
        logger.info(f"  Total features: {len(features_df.columns)}")
        logger.info(f"  Date range: {asof_date_str}")
        logger.info("="*80)
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error processing gold features: {str(e)}")
        raise
