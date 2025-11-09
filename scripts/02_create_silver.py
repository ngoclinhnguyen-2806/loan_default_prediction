"""
CLI Usage Examples:
  python /app/scripts/03_create_silver.py all
  python /app/scripts/03_create_silver.py all lms_loan_daily
  python /app/scripts/03_create_silver.py lms_loan_daily
"""

import os
import sys
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, regexp_replace, trim, when, year, month
from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
BRONZE_DIR = "/app/datamart/bronze"
SILVER_DIR = "/app/datamart/silver"
TABLES = ["lms_loan_daily", "feature_clickstream", "features_attributes", "features_financials"]

# -------------------------------------------------------------------------
# GENERIC HELPERS
# -------------------------------------------------------------------------
def cast_to_numeric(df, exclude=("Customer_ID", "snapshot_date")):
    """Clean and cast string columns that look numeric."""
    str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
    for c in [c for c in str_cols if c not in exclude]:
        clean = trim(regexp_replace(col(c), r"[^0-9.\\-]+", ""))
        df = df.withColumn(c, when(F.length(clean) == 0, None).otherwise(clean.cast(DoubleType())))
    return df

def read_bronze_table(spark, table):
    """Read parquet files from bronze directory."""
    path = os.path.join(BRONZE_DIR, table)
    if not os.path.exists(path):
        print(f"⚠️  {table} not found in bronze")
        return None
    df = spark.read.parquet(path)
    if "snapshot_date" in df.columns:
        df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))
    return df

def save_silver_table(df, table):
    """Write DataFrame to silver layer partitioned by year/month."""
    out_path = os.path.join(SILVER_DIR, table)
    os.makedirs(out_path, exist_ok=True)
    df.write.mode("overwrite").partitionBy("year", "month").parquet(out_path)
    print(f"✅ {table} → {out_path}")

# -------------------------------------------------------------------------
# TRANSFORMS
# -------------------------------------------------------------------------
def transform_clickstream(df):
    return cast_to_numeric(df)

def transform_attributes(df):
    pii = [c for c in ["SSN", "Name"] if c in df.columns]
    if pii:
        df = df.drop(*pii)
    df = cast_to_numeric(df)
    if "Age" in df.columns:
        df = df.withColumn("Age", when((col("Age") >= 18) & (col("Age") <= 85), col("Age")))
    return df

def transform_financials(df):
    exclude = ("Customer_ID","snapshot_date","Type_of_Loan","Credit_Mix","Payment_Behaviour","Credit_History_Age")
    df = cast_to_numeric(df, exclude)

    # Clean Payment_Behaviour
    valid = [
        "Low_spent_Small_value_payments","High_spent_Medium_value_payments",
        "Low_spent_Medium_value_payments","High_spent_Large_value_payments",
        "High_spent_Small_value_payments","Low_spent_Large_value_payments"
    ]
    df = df.withColumn(
        "Payment_Behaviour",
        when(~col("Payment_Behaviour").isin(valid), "Unknown").otherwise(col("Payment_Behaviour"))
    )

    # Parse Credit_History_Age
    yrs = F.regexp_extract(col("Credit_History_Age"), r"(\\d+)\\s*year", 1).cast(IntegerType())
    mos = F.regexp_extract(col("Credit_History_Age"), r"(\\d+)\\s*month", 1).cast(IntegerType())
    df = df.withColumn("Credit_History_Age_Year", F.coalesce(yrs, F.lit(0)) + F.coalesce(mos, F.lit(0))/12)
    df = df.withColumn("Credit_History_Age_Month", F.coalesce(yrs, F.lit(0))*12 + F.coalesce(mos, F.lit(0)))

    # DTI
    if all(c in df.columns for c in ["Outstanding_Debt","Annual_Income"]):
        df = df.withColumn("DTI", F.when(col("Annual_Income") > 0, col("Outstanding_Debt") / col("Annual_Income")))

    return df

def transform_loan(df):
    """Full loan transformation: enforce schema, MOB, installments missed, first_missed_date, dpd."""
    from pyspark.sql.types import FloatType
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for c, t in column_type_map.items():
        if c in df.columns:
            df = df.withColumn(c, col(c).cast(t))

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn(
        "installments_missed",
        F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())
    ).fillna(0)
    df = df.withColumn(
        "first_missed_date",
        F.when(col("installments_missed") > 0,
               F.add_months(col("snapshot_date"), -1 * col("installments_missed")))
         .cast(DateType())
    )
    df = df.withColumn(
        "dpd",
        F.when(col("overdue_amt") > 0.0,
               F.datediff(col("snapshot_date"), col("first_missed_date")))
         .otherwise(0)
         .cast(IntegerType())
    )
    return df

# -------------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------------
def process_table(spark, table):
    df = read_bronze_table(spark, table)
    if df is None:
        return
    df = df.withColumn("year", year("snapshot_date")).withColumn("month", month("snapshot_date"))
    if table == "feature_clickstream":
        df = transform_clickstream(df)
    elif table == "features_attributes":
        df = transform_attributes(df)
    elif table == "features_financials":
        df = transform_financials(df)
    elif table == "lms_loan_daily":
        df = transform_loan(df)
    save_silver_table(df, table)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create Silver Layer tables from Bronze data.")
    parser.add_argument("mode", choices=["all"], help="Specify 'all' to process all tables or a subset.")
    parser.add_argument("tables", nargs="*", help="Optional specific table names (e.g., lms_loan_daily).")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("silver_processing").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    os.makedirs(SILVER_DIR, exist_ok=True)

    if args.mode == "all":
        # If table names provided, process those only
        selected_tables = args.tables if args.tables else TABLES
        print(f"\n--- Processing Silver Layer for tables: {selected_tables} ---\n")
        for t in selected_tables:
            try:
                process_table(spark, t)
            except Exception as e:
                print(f"❌ {t} failed: {e}")
    else:
        print("Invalid mode. Only 'all' is supported currently.")

    spark.stop()
    print("\n--- Silver Layer Completed ---\n")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
