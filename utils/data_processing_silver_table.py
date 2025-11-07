# --- Standard libraries ---
import os
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- PySpark core & types ---
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.functions import col, regexp_replace, trim, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DoubleType, DateType, DecimalType

# --- CLI utilities (if you use them elsewhere) ---
import argparse

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# BASIC VALIDATION
# =============================================================================
def validate_customer_ids(df, table_name):
    """
    Ensure Customer_ID exists; warn on nulls; log unique count.
    """
    if "Customer_ID" not in df.columns:
        error_msg = f"{table_name}: Customer_ID column not found"
        logger.error(error_msg)
        raise ValueError(error_msg)

    null_customers = df.filter(col("Customer_ID").isNull()).count()
    if null_customers > 0:
        logger.warning(f"{table_name}: {null_customers} rows with null Customer_ID")

    unique_customers = df.select("Customer_ID").distinct().count()
    logger.info(f"{table_name}: {unique_customers} unique customers")


def dedup_on(df, keys, order_cols=None):
    """
    Keep the latest row per key. If order_cols provided, prefer the newest
    (DESC, NULLS LAST); otherwise simple dropDuplicates(keys).
    """
    if order_cols:
        w = Window.partitionBy(*keys).orderBy(*[F.col(c).desc_nulls_last() for c in order_cols])
        return df.withColumn("__rn", F.row_number().over(w)).filter(F.col("__rn")==1).drop("__rn")
    return df.dropDuplicates(keys)

# =============================================================================
# HELPERS
# =============================================================================
def load_bronze_feature_table(snapshot_date_str, bronze_directory, table_name, spark):
    """
    Load bronze feature table and apply common preprocessing.
    Returns Spark DataFrame or None if not found.
    """
    try:
        bronze_table_directory = os.path.join(bronze_directory, table_name + "/")
        partition_name = f"bronze_{table_name}_{snapshot_date_str.replace('-','_')}.csv"
        filepath = os.path.join(bronze_table_directory, partition_name)

        if not os.path.exists(filepath):
            logger.warning(f"Bronze file not found: {filepath}")
            return None

        df = spark.read.csv(filepath, header=True, inferSchema=True)
        logger.info(f"Loaded {table_name} from {filepath}: {df.count()} rows")

        # Drop unnamed index column if present
        if "" in df.columns:
            df = df.drop("")
        if "_c0" in df.columns:
            df = df.drop("_c0")

        # Basic sanity: id/date types
        validate_customer_ids(df, table_name)
        if "Customer_ID" in df.columns:
            df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
        if "snapshot_date" in df.columns:
            df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

        return df
    except Exception as e:
        logger.error(f"Error loading bronze table {table_name}: {str(e)}")
        raise


def save_silver_table(df, snapshot_date_str, silver_directory, table_name):
    """
    Save DataFrame to silver layer as parquet (overwrite).
    """
    try:
        table_silver_directory = os.path.join(silver_directory, table_name + "/")
        if not os.path.exists(table_silver_directory):
            os.makedirs(table_silver_directory)
            logger.info(f"Created directory: {table_silver_directory}")

        partition_name = f"silver_{table_name}_{snapshot_date_str.replace('-','_')}.parquet"
        filepath = os.path.join(table_silver_directory, partition_name)

        df.write.mode("overwrite").parquet(filepath)
        logger.info(f"Saved {table_name} to {filepath}")
    except Exception as e:
        logger.error(f"Error saving silver table {table_name}: {str(e)}")
        raise


def cast_to_numeric(df, exclude=("Customer_ID", "snapshot_date"), numeric_threshold=0.9):
    """
    Auto-detect numeric-looking string columns, clean, and cast to Integer/Double.
    """
    try:
        string_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]
        candidates = [c for c in string_cols if c not in exclude]

        for c in candidates:
            cleaned = regexp_replace(col(c), r"[^0-9\.\-]+", "")
            cleaned = trim(cleaned)

            cast_col = when(F.length(cleaned) == 0, None).otherwise(cleaned).cast(DoubleType())
            tmp = f"__num_{c}"

            ratio = df.withColumn(tmp, cast_col) \
                      .select((F.count(tmp) / F.count(F.lit(1))).alias("ratio")) \
                      .collect()[0]["ratio"]

            if ratio is not None and ratio >= numeric_threshold:
                df_with = df.withColumn(tmp, cast_col)
                max_frac = df_with.select(
                    F.max(F.abs(col(tmp) - F.floor(col(tmp)))).alias("max_frac")
                ).collect()[0]["max_frac"]

                is_integer = (max_frac is None) or (float(max_frac) == 0.0)
                target_type = IntegerType() if is_integer else DoubleType()

                df = df_with.drop(c).withColumn(c, col(tmp).cast(target_type)).drop(tmp)
                logger.info(
                    "Auto-cast '%s' -> %s (cleaned non-numeric chars)",
                    c, "Integer" if is_integer else "Double"
                )
            else:
                logger.info(
                    "Kept '%s' as string (only %.2f%% numeric after cleaning)",
                    c, (ratio or 0.0) * 100
                )

        return df
    except Exception as e:
        logger.error(f"Error in transformation: {str(e)}")
        raise


def upcast_floats_to_double(df):
    for f in df.schema.fields:
        if isinstance(f.dataType, (FloatType, DecimalType)):
            df = df.withColumn(f.name, col(f.name).cast(DoubleType()))
    return df
        
# =============================================================================
# TRANSFORMATIONS
# =============================================================================

def transform_clickstream(df):
    try:
        # numeric sweep (protect ID/date)
        df = cast_to_numeric(df, exclude=("Customer_ID", "snapshot_date"))

        # as-of date for leakage control
        if "snapshot_date" in df.columns:
            df = df.withColumn("asof_date", col("snapshot_date"))

        # deduplicate at grain: Customer_ID + snapshot_date (prefer updated_at if present)
        keys = ["Customer_ID", "snapshot_date"]
        order_cols = [c for c in ["updated_at", "snapshot_date"] if c in df.columns]
        df = dedup_on(df, keys, order_cols=order_cols if order_cols else None)

        return df
    except Exception as e:
        logger.error(f"Error in clickstream transformation: {str(e)}")
        raise


def transform_attributes(df):
    try:
        # optional PII removal
        pii_cols = [c for c in ["SSN", "Name"] if c in df.columns]
        if pii_cols:
            df = df.drop(*pii_cols)

        # numeric sweep
        df = cast_to_numeric(df, exclude=("Customer_ID", "snapshot_date"))
        df = upcast_floats_to_double(df)

        # Age cleaning: keep only [18, 85], else set to null
        if "Age" in df.columns:
            df = df.withColumn(
                "age_clean",
                F.when((F.col("Age") >= 18) & (F.col("Age") <= 85), F.col("Age").cast("int"))
                 .otherwise(F.lit(None).cast("int"))
            )

        # as-of date
        if "snapshot_date" in df.columns:
            df = df.withColumn("asof_date", col("snapshot_date"))

        # deduplicate at grain
        keys = ["Customer_ID", "snapshot_date"]
        order_cols = [c for c in ["updated_at", "snapshot_date"] if c in df.columns]
        df = dedup_on(df, keys, order_cols=order_cols if order_cols else None)

        return df
        
    except Exception as e:
        logger.error(f"Error in attributes transformation: {str(e)}")
        raise



def transform_financials(df):
    try:
        # 1) numeric sweep (protect categoricals/text)
        exclude = ("Customer_ID","snapshot_date","Type_of_Loan","Credit_Mix","Payment_Behaviour","Credit_History_Age")
        df = cast_to_numeric(df, exclude=exclude)
        df = upcast_floats_to_double(df)

        # 2) normalize Payment_Behaviour -> 'Unknown'
        valid_behaviours = [
            "Low_spent_Small_value_payments",
            "High_spent_Medium_value_payments",
            "Low_spent_Medium_value_payments",
            "High_spent_Large_value_payments",
            "High_spent_Small_value_payments",
            "Low_spent_Large_value_payments",
        ]
        bad_tokens = ["na","n/a","none","null","-","?","unknown","undefined","nan"]
        raw = F.trim(F.col("Payment_Behaviour"))
        only_letters_underscores = F.regexp_replace(raw, r"[^A-Za-z_]", "")
        df = df.withColumn(
            "Payment_Behaviour",
            F.when(
                raw.isNull()
                | (F.length(raw) == 0)
                | F.lower(raw).isin(bad_tokens)
                | (raw != only_letters_underscores)
                | (~raw.isin(valid_behaviours)),
                F.lit("Unknown"),
            ).otherwise(raw)
        )

        # 3) parse Credit_History_Age -> years/months
        yrs  = F.regexp_extract(F.col("Credit_History_Age"), r"(?i)(\d+)\s*year", 1)
        mos  = F.regexp_extract(F.col("Credit_History_Age"), r"(?i)(\d+)\s*month", 1)
        yrsN = F.when(F.length(yrs)==0, None).otherwise(yrs.cast(IntegerType()))
        mosN = F.when(F.length(mos)==0, None).otherwise(mos.cast(IntegerType()))
        has_any = yrsN.isNotNull() | mosN.isNotNull()
        yrsNZ = F.coalesce(yrsN, F.lit(0))
        mosNZ = F.coalesce(mosN, F.lit(0))
        df = df.withColumn(
            "Credit_History_Age_Year",
            F.when(has_any, yrsNZ.cast(DoubleType()) + mosNZ.cast(DoubleType())/F.lit(12.0)).otherwise(F.lit(None).cast(DoubleType()))
        ).withColumn(
            "Credit_History_Age_Month",
            F.when(has_any, yrsNZ*F.lit(12) + mosNZ).otherwise(F.lit(None).cast(IntegerType()))
        )

        # 4) DTI = Outstanding_Debt / Annual_Income
        if all(c in df.columns for c in ["Outstanding_Debt","Annual_Income"]):
            df = df.withColumn(
                "DTI",
                F.when((col("Annual_Income") > 0) & col("Outstanding_Debt").isNotNull(),
                       col("Outstanding_Debt") / col("Annual_Income")
                ).otherwise(None).cast(DoubleType())
            )

        # 5) multi-hot for Type_of_Loan
        if "Type_of_Loan" in df.columns:
            # normalize & split tokens (remove spaces then split by comma)
            toks = F.split(F.regexp_replace(F.coalesce(col("Type_of_Loan"), F.lit("")), r"\s+", ""), ",")
            loan_types = [
                "AutoLoan","Credit-Builder","PersonalLoan","HomeEquity",
                "Mortgage","StudentLoan","DebtConsolidation"
            ]
            for t in loan_types:
                df = df.withColumn(f"loan_type__{t}", F.array_contains(toks, F.lit(t)).cast("int"))
            # optional: count how many types per row
            df = df.withColumn(
                "loan_type_count",
                sum([F.col(f"loan_type__{t}") for t in loan_types])
            )

        # 6) as-of date
        if "snapshot_date" in df.columns:
            df = df.withColumn("asof_date", col("snapshot_date"))

        # 7) deduplicate at grain
        keys = ["Customer_ID", "snapshot_date"]
        order_cols = [c for c in ["updated_at", "snapshot_date"] if c in df.columns]
        df = dedup_on(df, keys, order_cols=order_cols if order_cols else None)

        logger.info("Financials: numeric sweep + behaviour cleanup + credit age parsed + DTI + loan multi-hot + dedup + asof.")
        return df
        
    except Exception as e:
        logger.error(f"Error in financials transformation: {str(e)}")
        raise
        

def transform_loan(df):
    try:
        # enforce key id/date types explicitly
        type_map = {
            "loan_id": StringType(),
            "Customer_ID": StringType(),
            "loan_start_date": DateType(),
            "snapshot_date": DateType(),
        }
        for c, t in type_map.items():
            if c in df.columns:
                df = df.withColumn(c, col(c).cast(t))

        # numeric sweep for all other numeric-like fields
        df = cast_to_numeric(df, exclude=("Customer_ID", "snapshot_date", "loan_id", "loan_start_date"))
        df = upcast_floats_to_double(df)

        # derived: MOB (= installment_num), installments_missed, first_missed_date, DPD
        df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
        safe_due = F.when((col("due_amt").isNotNull()) & (col("due_amt") != 0), col("due_amt"))
        inst_missed = F.ceil(col("overdue_amt") / safe_due)
        df = df.withColumn("installments_missed", F.when(inst_missed.isNotNull(), inst_missed).otherwise(0).cast(IntegerType()))
        df = df.withColumn(
            "first_missed_date",
            F.when(col("installments_missed") > 0,
                   F.add_months(col("snapshot_date"), -1 * col("installments_missed"))
            ).cast(DateType())
        )
        df = df.withColumn(
            "dpd",
            F.when(col("overdue_amt") > 0.0,
                   F.datediff(col("snapshot_date"), col("first_missed_date"))
            ).otherwise(0).cast(IntegerType())
        )

        # as-of date
        if "snapshot_date" in df.columns:
            df = df.withColumn("asof_date", col("snapshot_date"))

        # deduplicate at grain (loan_id + snapshot_date is typical)
        keys = ["loan_id", "snapshot_date"] if "loan_id" in df.columns else ["Customer_ID", "snapshot_date"]
        order_cols = [c for c in ["updated_at", "snapshot_date"] if c in df.columns]
        df = dedup_on(df, keys, order_cols=order_cols if order_cols else None)

        logger.info("Loan: types enforced + numeric sweep + MOB/DPD + dedup + asof.")
        return df
    except Exception as e:
        logger.error(f"Error in loan transformation: {str(e)}")
        raise

# =============================================================================
# PROCESSORS
# =============================================================================

def process_silver_clickstream_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Bronze → Silver for clickstream.
    """
    try:
        table_name = "feature_clickstream"
        logger.info(f"Starting clickstream processing for {snapshot_date_str}")

        df = load_bronze_feature_table(snapshot_date_str, bronze_directory, table_name, spark)
        if df is None:
            return None

        df = transform_clickstream(df)
        save_silver_table(df, snapshot_date_str, silver_directory, table_name)

        logger.info(f"Completed clickstream processing for {snapshot_date_str}")
        return df
    except Exception as e:
        logger.error(f"Error processing clickstream table: {str(e)}")
        raise


def process_silver_attributes_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Bronze → Silver for attributes.
    """
    try:
        table_name = "features_attributes"
        logger.info(f"Starting attributes processing for {snapshot_date_str}")

        df = load_bronze_feature_table(snapshot_date_str, bronze_directory, table_name, spark)
        if df is None:
            return None

        df = transform_attributes(df)
        save_silver_table(df, snapshot_date_str, silver_directory, table_name)

        logger.info(f"Completed attributes processing for {snapshot_date_str}")
        return df
    except Exception as e:
        logger.error(f"Error processing attributes table: {str(e)}")
        raise


def process_silver_financials_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Bronze → Silver for financials.
    """
    try:
        table_name = "features_financials"
        logger.info(f"Starting financials processing for {snapshot_date_str}")

        df = load_bronze_feature_table(snapshot_date_str, bronze_directory, table_name, spark)
        if df is None:
            return None

        df = transform_financials(df)
        save_silver_table(df, snapshot_date_str, silver_directory, table_name)

        logger.info(f"Completed financials processing for {snapshot_date_str}")
        return df
    except Exception as e:
        logger.error(f"Error processing financials table: {str(e)}")
        raise


def process_silver_loan_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """
    Load bronze loan CSV, apply transform_loan, save to silver parquet.
    """
    try:
        table_name = "lms_loan_daily"
        silver_dir = os.path.join(silver_directory, "loan_daily/")
        if not os.path.exists(silver_dir):
            os.makedirs(silver_dir)
            logger.info(f"Created directory: {silver_dir}")

        bronze_dir = os.path.join(bronze_directory, "lms_loan_daily/")
        partition_name = f"bronze_lms_loan_daily_{snapshot_date_str.replace('-','_')}.csv"
        filepath = os.path.join(bronze_dir, partition_name)

        if not os.path.exists(filepath):
            logger.warning(f"Bronze file not found: {filepath}")
            return None

        df = spark.read.csv(filepath, header=True, inferSchema=True)
        logger.info(f"Loaded {table_name} from {filepath}: {df.count()} rows")

        # Keep light sanity check only
        validate_customer_ids(df, table_name)

        df = transform_loan(df)

        out_file = os.path.join(silver_dir, f"silver_loan_daily_{snapshot_date_str.replace('-','_')}.parquet")
        df.write.mode("overwrite").parquet(out_file)
        logger.info(f"Saved loan_daily to {out_file}")
        return df
    except Exception as e:
        logger.error(f"Error processing loan table: {str(e)}")
        raise
        

# =============================================================================
# ORCHESTRATION
# =============================================================================
def get_table_processor(table_name):
    """
    Map table names to their processing functions.
    """
    table_function_mapping = {
        "lms_loan_daily": process_silver_loan_table,
        "feature_clickstream": process_silver_clickstream_table,
        "features_attributes": process_silver_attributes_table,
        "features_financials": process_silver_financials_table,
    }
    processor = table_function_mapping.get(table_name)
    if processor is None:
        error_msg = f"Unknown table name: {table_name}. Valid options: {list(table_function_mapping.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return processor


def process_silver_table(snapshot_date_str, bronze_directory, silver_directory, spark, table_name=None):
    """
    Process one or all tables for a given snapshot date.
    """
    try:
        if table_name:
            logger.info(f"Processing single table: {table_name} for {snapshot_date_str}")
            processor = get_table_processor(table_name)
            result = processor(snapshot_date_str, bronze_directory, silver_directory, spark)
            logger.info(f"Successfully processed {table_name}")
            return result
        else:
            all_tables = ["lms_loan_daily", "feature_clickstream", "features_attributes", "features_financials"]
            results = {}

            logger.info("=" * 60)
            logger.info(f"Processing Silver Tables for {snapshot_date_str}")
            logger.info("=" * 60)

            for table in all_tables:
                logger.info(f"Processing {table}...")
                try:
                    processor = get_table_processor(table)
                    df = processor(snapshot_date_str, bronze_directory, silver_directory, spark)
                    results[table] = df
                    logger.info(f"✓ {table} completed")
                except Exception as e:
                    logger.error(f"✗ Error processing {table}: {str(e)}")
                    results[table] = None  # continue with others

            logger.info("=" * 60)
            logger.info(f"Silver layer processing completed for {snapshot_date_str}")
            logger.info("=" * 60)
            return results
    except Exception as e:
        logger.error(f"Critical error in process_silver_table: {str(e)}")
        raise
