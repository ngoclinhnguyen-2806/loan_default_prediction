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
import re

from pyspark.sql.functions import col, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

_DATE_IN_NAME = re.compile(r".*?(\d{4})_(\d{2})_(\d{2})\.(?:parquet|csv)$", re.IGNORECASE)

def _infer_snapshot_from_name(filename: str):
    m = _DATE_IN_NAME.match(filename)
    if not m:
        return None
    y, mth, d = m.groups()
    return f"{y}-{mth}-{d}"


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_features_gold_table(silver_attr_directory, silver_fin_directory, silver_clickstream_directory, gold_feature_store_directory, spark):

    #---------------
    # ATTRIBUTES
    #---------------
    parquet_files = glob.glob(os.path.join(silver_attr_directory, "*.parquet"))
    df_attr = spark.read.parquet(*parquet_files)
    df_attr = df_attr.withColumn("is_occu_known", F.when(F.trim(F.col("Occupation")).rlike("(?i)^unknown$"), 0).otherwise(1))
    df_attr = df_attr.withColumn("age_band", when(col("Age") == 0, "Unknown") \
                                            .when(col("Age") < 25, "18-24") \
                                            .when((col("Age") >= 25) & (col("Age") < 35), "25-34") \
                                            .when((col("Age") >= 35) & (col("Age") < 45), "35-44") \
                                            .when((col("Age") >= 45) & (col("Age") < 55), "45-54") \
                                            .otherwise("55+"))
    
    for band in ["Unknown", "18-24", "25-34", "35-44", "45-54", "55+"]:
            df_attr = df_attr.withColumn(f"age_band_{band.replace('-','_').replace('+','')}",
                                           when(col("age_band")==band, 1).otherwise(0))   
        
    df_attr = df_attr.drop("Age", "Name", "SSN", "Occupation","age_band")

    #---------------
    # FINANCIALS
    #---------------
    parquet_files = glob.glob(os.path.join(silver_fin_directory, "*.parquet"))
    df_fin = spark.read.parquet(*parquet_files)

    fin_cols_keep = [
                "Annual_Income","Outstanding_Debt","Payment_Behaviour","Credit_Mix",
                "Type_of_Loan","Credit_History_Age","Num_of_Loan","Num_of_Delayed_Payment",
                "Delay_from_due_date","snapshot_date"
            ]
    fin_pre = df_fin.select("Customer_ID", *fin_cols_keep)

    # Capacity
    features = fin_pre.withColumn("DTI",
                    when(col("Annual_Income") > 0,
                         col("Outstanding_Debt") / col("Annual_Income")))
    features = features.withColumn("log_Annual_Income",
                    when(col("Annual_Income") > 0, F.log(col("Annual_Income"))))

    # Behaviorals
    pb_vals = [
                "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
                "Low_spent_Medium_value_payments","Low_spent_Small_value_payments",
                "High_spent_Medium_value_payments","High_spent_Large_value_payments"
            ]
    for v in pb_vals:
        features = features.withColumn(f"Payment_Behaviour_{v.replace(' ','_').replace('-','_')}",
                                       when(col("Payment_Behaviour")==v,1).otherwise(0))
    for v in ["Standard","Good","Bad"]:
        features = features.withColumn(f"Credit_Mix_{v}", when(col("Credit_Mix")==v,1).otherwise(0))
    loan_types = [
        "Auto Loan","Credit-Builder Loan","Personal Loan","Home Equity Loan",
        "Mortgage Loan","Student Loan","Debt Consolidation Loan","Payday Loan"
    ]
    for v in loan_types:
        features = features.withColumn(f"Type_of_Loan_{v.replace(' ','_').replace('-','_')}",
                                       when(col("Type_of_Loan").contains(v),1).otherwise(0))

    # DROP ORIGINAL CATEGORICAL COLUMNS
    df_fin = features.drop("Payment_Behaviour", "Credit_Mix", "Type_of_Loan")
    
    # Calculate null percentages
    rows = df_fin.count()

    nulls = df_fin.select([
        (F.count(F.when(F.col(c).isNull(), c)) / rows * 100).alias(c)
        for c in df_fin.columns
    ])
    
    # Collect null percentages into a dict
    nulls_dict = nulls.collect()[0].asDict()
    
    # Get columns to drop
    cols_to_drop = [col for col, pct in nulls_dict.items() if pct > 5]
    
    # Drop columns from DataFrame
    df_fin = df_fin.drop(*cols_to_drop)
    print("Dropped columns:", cols_to_drop)

    # Drop the rest of the NAs
    df_fin = df_fin.dropna()   

    #---------------
    # CLICKSTREAM
    #---------------
    parquet_files = glob.glob(os.path.join(silver_clickstream_directory, "*.parquet"))
    df_click = spark.read.parquet(*parquet_files)

    app_store = '/app/datamart/gold/application_store'
    df_app = spark.read.parquet(app_store)
    df_app= df_app.drop("loan_amt","tenure")

    # Get only clicks before or on application date
    df_click_filtered = df_click.join(df_app, on='Customer_ID', how='right') \
                                .filter(F.col("snapshot_date") <= F.col("application_date"))  
    
    # Sum all clicks before or on application date
    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    agg_exprs = [F.sum(F.col(c)).alias(f"{c}_sum_all") for c in feature_cols]
    df_click_sum = (
        df_click_filtered.groupBy("Customer_ID", "application_date")
                         .agg(*agg_exprs)
    )

    # Clicks on application date
    df_click_ondate = df_click_filtered.filter(F.col("snapshot_date") == F.col("application_date"))
    df_click_ondate = df_click_ondate.drop("application_date")

    # Join all clicks
    df_click_features = df_click_sum.join(df_click_ondate, on="Customer_ID", how='left')
    
    #---------------
    # JOIN ALL FEATURES
    #---------------
    # Drop columns to avoid duplicated column names
    features = features.drop("snapshot_date")
    df_attr = df_attr.drop("snapshot_date")
    df_click_features = df_click_features.drop("snapshot_date", "application_date")
    
    # Join all
    df_features = df_app.join(df_fin, how="left", on=["Customer_ID"]) \
                        .join(df_attr, how="left", on=["Customer_ID"]) \
                        .join(df_click_features, how="left", on=["Customer_ID"])
    
    # Retain column names to not affect downstream tasks
    df_features = df_features.withColumn("snapshot_date", col("application_date"))
    print("Added column snapshot_date based on application_date")
    df_features.show(5)

    # Final null handling
    numeric_cols = [f.name for f in df_features.schema.fields 
                    if f.dataType.typeName() in ['integer', 'long', 'double', 'float']]    
    fill_dict = {col: 0 for col in numeric_cols}
    df_features = df_features.fillna(fill_dict)

    gold_feature_store_directory = "/app/datamart/gold/feature_store/"
    silver_attr_directory = "/app/datamart/silver/features_attributes/"
    silver_fin_directory = "/app/datamart/silver/features_financials/"
    silver_clickstream_directory = "/app/datamart/silver/feature_clickstream/"

    # --- Write by snapshot_date ---
    df_features.write.mode("overwrite") \
               .partitionBy("snapshot_date") \
               .parquet(gold_feature_store_directory)
    print("Written to gold/datamart/feature_store")

    return df_features