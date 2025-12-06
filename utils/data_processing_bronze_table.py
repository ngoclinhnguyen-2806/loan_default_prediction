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


def process_bronze_table(csv_file_path, bronze_directory, spark, prefix="bronze_table"):
    os.makedirs(bronze_directory, exist_ok=True)

    # Read once
    df_base = spark.read.csv(csv_file_path, header=True, inferSchema=True)

    # List of dates as strings
    dates_str_lst = (
        df_base
        .select(F.col("snapshot_date").cast("string"))
        .distinct()
        .orderBy("snapshot_date")
        .rdd.flatMap(lambda x: x)
        .collect()
    )

    for snapshot_date_str in dates_str_lst:
        # IMPORTANT: filter the base df, don't overwrite it
        df_snapshot = df_base.filter(F.col("snapshot_date") == snapshot_date_str)

        outpath = os.path.join(
            bronze_directory,
            f"{prefix}_{snapshot_date_str.replace('-', '_')}.csv"
        )
        df_snapshot.toPandas().to_csv(outpath, index=False)

    print(f"Saved {len(dates_str_lst)} bronze files to {bronze_directory}")
    return df_base