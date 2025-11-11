# gold_features_processing_optimized.py
# Short, production-friendly feature builder for application-time default prediction

import os, glob, logging
from typing import List, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lit, when, datediff, avg, stddev, sum as _sum, max as _max

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gold_features")

# ----------------------------
# I/O
# ----------------------------

log = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# SIMPLE LOAD SILVER TABLE FUNCTION
# -------------------------------------------------------------------------
def _load_table(spark, table_name, snapshot_date=None):
    """
    Load a silver table as Spark DataFrame.
    Args:
        spark: SparkSession
        table_name (str): name of silver table (e.g. "features_financials")
        snapshot_date (str, optional): e.g. "2023-01-01".
                                       If None, loads and unions all snapshots.
    Returns:
        Spark DataFrame
    """
    import os

    silver_table_dir = os.path.join("/app/datamart/silver", table_name)

    if snapshot_date:
        path = os.path.join(
            silver_table_dir,
            f"silver_{table_name}_{snapshot_date.replace('-', '_')}.parquet"
        )
        print(f"📂 Loading snapshot: {path}")
        return spark.read.parquet(path)

    else:
        # list all parquet files for this table
        all_paths = sorted(
            [
                os.path.join(silver_table_dir, d)
                for d in os.listdir(silver_table_dir)
                if d.startswith(f"silver_{table_name}_") and d.endswith(".parquet")
            ]
        )

        print(f"📦 Loading {len(all_paths)} snapshots from {silver_table_dir}")

        if not all_paths:
            raise FileNotFoundError(f"No parquet files found for table {table_name} at {silver_table_dir}")

        # read and union all snapshots
        dfs = [spark.read.parquet(p) for p in all_paths]
        df_all = dfs[0]
        for df_next in dfs[1:]:
            df_all = df_all.unionByName(df_next, allowMissingColumns=True)
        return df_all

# ----------------------------
# HELPERS
# ----------------------------
def _latest_pre_app(df: DataFrame, apps: DataFrame, keys: List[str]) -> DataFrame:
    """Join df to apps on Customer_ID, keep rows with snapshot_date < application_date and take latest snapshot per customer."""
    pre = (
        df.join(apps.select("Customer_ID", "application_date"), "Customer_ID", "inner")
          .filter(col("snapshot_date") <= col("application_date"))
    )
    w = Window.partitionBy("Customer_ID").orderBy(col("snapshot_date").desc())
    return (pre.withColumn("rn", F.row_number().over(w))
               .filter(col("rn") == 1)
               .drop("rn")
               .select("Customer_ID", *keys))

def _click_agg(click: DataFrame, apps: DataFrame, horizon_days: int) -> DataFrame:
    pre = (
        click.join(apps.select("Customer_ID", "application_date"), "Customer_ID", "inner")
             .filter(col("snapshot_date") <= col("application_date"))
             .filter(datediff(col("application_date"), col("snapshot_date")).between(0, horizon_days))
    )
    fe_cols = [c for c in pre.columns if c.startswith("fe_")]
    if not fe_cols:
        return apps.select("Customer_ID").dropDuplicates(["Customer_ID"])
    
    # ✅ Fill nulls in fe_ columns before aggregation
    fill_dict = {c: 0 for c in fe_cols}
    pre = pre.fillna(fill_dict)
    
    aggs = []
    for c in fe_cols:
        aggs += [_sum(c).alias(f"{c}_sum_{horizon_days}d"),
                 avg(c).alias(f"{c}_mean_{horizon_days}d"),
                 stddev(c).alias(f"{c}_std_{horizon_days}d")]
    
    result = pre.groupBy("Customer_ID").agg(*aggs)
    
    # ✅ Fill nulls in aggregated columns (for stddev which can still return null)
    agg_cols = [f"{c}_sum_{horizon_days}d" for c in fe_cols] + \
               [f"{c}_mean_{horizon_days}d" for c in fe_cols] + \
               [f"{c}_std_{horizon_days}d" for c in fe_cols]
    result = result.fillna({c: 0 for c in agg_cols})
    
    return result

def _delinquency_agg(fin_pre_joined: DataFrame) -> DataFrame:
    # fin_pre_joined must already be joined to apps and filtered to < application_date (multiple snapshots per cust)
    base = fin_pre_joined
    def roll(d):
        f = base.filter(datediff(col("application_date"), col("snapshot_date")).between(0, d))
        return f.groupBy("Customer_ID").agg(_sum("Num_of_Delayed_Payment").alias(f"Num_of_Delayed_Payment_{d//30 if d>=30 else d}m"))
    agg3m  = roll(90)
    agg6m  = roll(180)
    agg12m = (base.filter(datediff(col("application_date"), col("snapshot_date")).between(0, 365))
                   .groupBy("Customer_ID")
                   .agg(_sum("Num_of_Delayed_Payment").alias("Num_of_Delayed_Payment_12m"),
                        _max("Num_of_Delayed_Payment").alias("max_dpd_prior")))
    out = (agg3m.join(agg6m, "Customer_ID", "outer")
                .join(agg12m, "Customer_ID", "outer")
                .fillna({"Num_of_Delayed_Payment_3m":0, "Num_of_Delayed_Payment_6m":0,
                         "Num_of_Delayed_Payment_12m":0, "max_dpd_prior":0})
                .withColumn("ever_30dpd_prior", when(col("max_dpd_prior") >= 30, 1).otherwise(0)))
    return out

# ----------------------------
# FEATURE BUILDER
# ----------------------------
def build_features(apps: DataFrame, SILVER_DIR: str, asof_date: str, spark: SparkSession) -> DataFrame:
    fin = _load_table(spark, "features_financials")
    att = _load_table(spark, "features_attributes")
    clk = _load_table(spark, "feature_clickstream")

    features = apps

    # ---- FINANCIALS ----
    if fin is not None:
        fin_cols_keep = [
            "Annual_Income","Outstanding_Debt","Payment_Behaviour","Credit_Mix",
            "Type_of_Loan","Credit_History_Age","Num_of_Loan","Num_of_Delayed_Payment",
            "Delay_from_due_date","snapshot_date"
        ]

        fin_pre = fin.select("Customer_ID", *fin_cols_keep)
        fin_latest = _latest_pre_app(
            fin_pre, features,
            keys=["Annual_Income","Outstanding_Debt","Payment_Behaviour","Credit_Mix",
                  "Type_of_Loan","Credit_History_Age","Num_of_Loan"]
        )
        features = features.join(fin_latest, "Customer_ID", "left")

        # Capacity
        features = features.withColumn("DTI",
                        when(col("Annual_Income") > 0,
                             col("Outstanding_Debt") / col("Annual_Income")))
        features = features.withColumn("log_Annual_Income",
                        when(col("Annual_Income") > 0, F.log(col("Annual_Income"))))

        # Behavioral one-hots
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

        # ✅ DROP ORIGINAL CATEGORICAL COLUMNS
        features = features.drop("Payment_Behaviour", "Credit_Mix", "Type_of_Loan")

        # Credit depth
        features = features.withColumn(
            "Credit_History_Age_Year",
            F.regexp_extract("Credit_History_Age", r"(\d+)\s*Years", 1).cast("int")
        ).withColumn(
            "Credit_History_Age_Month",
            F.regexp_extract("Credit_History_Age", r"(\d+)\s*Months", 1).cast("int")
        ).withColumnRenamed("Num_of_Loan","Num_of_Loan_active") \
         .drop("Credit_History_Age")

        # Delinquency (multi-snapshot pre-app; reuse pre join)
        fin_pre_multi = (
            fin_pre.join(features.select("Customer_ID","application_date"), "Customer_ID", "inner")
                   .filter(col("snapshot_date") <= col("application_date"))
        )
        delin = _delinquency_agg(fin_pre_multi)
        features = features.join(delin, "Customer_ID", "left")
    else:
        log.warning("features_financials missing")

    # ---- ATTRIBUTES (single latest snapshot per customer) ----
    if att is not None:
        att_keep = ["Age","Occupation","snapshot_date"]
        att_latest = _latest_pre_app(att.select("Customer_ID", *att_keep), features, keys=["Age","Occupation"])
        features = (features.join(att_latest, "Customer_ID", "left")
                    .withColumn("age_band",
                        when(col("Age") < 25, "18-24")
                        .when((col("Age") >= 25) & (col("Age") < 35), "25-34")
                        .when((col("Age") >= 35) & (col("Age") < 45), "35-44")
                        .when((col("Age") >= 45) & (col("Age") < 55), "45-54")
                        .otherwise("55+")))
        # ✅ ONE-HOT ENCODE age_band
        for band in ["18-24", "25-34", "35-44", "45-54", "55+"]:
            features = features.withColumn(f"age_band_{band.replace('-','_').replace('+','')}",
                                          when(col("age_band")==band, 1).otherwise(0))
        
        # ✅ DROP CATEGORICAL COLUMNS
        features = features.drop("age_band", "Occupation")
    else:
        log.warning("features_attributes missing")

    # ---- CLICKSTREAM (duplicate Customer_ID+snapshot_date handled by groupBy) ----
    if clk is not None:
        agg7  = _click_agg(clk, features, 7)
        agg30 = _click_agg(clk, features, 30)
        features = features.join(agg7, "Customer_ID", "left").join(agg30, "Customer_ID", "left")
    else:
        log.warning("feature_clickstream missing")

    # ---- APPLICATION-BASED FEATURES ----
    monthly_rate = lit(0.12/12.0)
    features = features.withColumn(
        "estimated_EMI",
        (col("loan_amt") * monthly_rate * F.pow(1 + monthly_rate, col("tenure"))) /
        (F.pow(1 + monthly_rate, col("tenure")) - 1)
    )
    features = features.withColumn(
        "EMI_to_income",
        when(col("Annual_Income") > 0, col("estimated_EMI") / (col("Annual_Income")/12.0))
    )
    features = features.withColumn("requested_amount", col("loan_amt")) \
                       .withColumn("requested_tenure", col("tenure"))

    # Replace any remaining nulls in DTI, log_Annual_Income, EMI_to_income with 0
    features = features.fillna({
        "DTI": 0,
        "log_Annual_Income": 0,
        "EMI_to_income": 0,
        "Credit_History_Age_Year": 0,
        "Credit_History_Age_Month": 0
    })

    # ---- IDENTIFIERS & DEDUP KEYS ----
    features = features.withColumn("snapshot_date", col("application_date").cast("date"))

    # Enforce UNIQUE (Customer_ID, snapshot_date) across mart
    wkey = Window.partitionBy("Customer_ID", "snapshot_date").orderBy(col("application_date").desc())
    features = (features.withColumn("rn_dedup", F.row_number().over(wkey))
                       .filter(col("rn_dedup") == 1)
                       .drop("rn_dedup"))

    # ---- NULL HANDLING ----
    # Fill numeric nulls with 0, except for key identifier columns
    exclude_from_fill = ["Customer_ID", "loan_id", "application_date", "snapshot_date"]
    numeric_cols = [f.name for f in features.schema.fields 
                    if f.dataType.typeName() in ['integer', 'long', 'double', 'float']
                    and f.name not in exclude_from_fill]
    
    fill_dict = {col: 0 for col in numeric_cols}
    features = features.fillna(fill_dict)

    # Enforce UNIQUE (Customer_ID, snapshot_date) across mart
    wkey = Window.partitionBy("Customer_ID", "snapshot_date").orderBy(col("application_date").desc())
    features = (features.withColumn("rn_dedup", F.row_number().over(wkey))
                       .filter(col("rn_dedup") == 1)
                       .drop("rn_dedup"))

    return features

# ----------------------------
# EXAMPLE USAGE (comment out in production entrypoints)
# ----------------------------
# spark = SparkSession.builder.getOrCreate()
# apps_train = make_training_apps(lms_loan_daily_df, "2025-12-31")
# feats = build_features(apps_train, "/data/silver", "2025-12-31", spark)
# feats.write.mode("overwrite").parquet("/data/gold/features_app_level")
