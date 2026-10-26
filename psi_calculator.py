# -*- coding: utf-8 -*-
"""
PSI (Population Stability Index) Calculator (PySpark / Hive)

Functions:
  1. Read detail & baseline from Hive via spark.sql()
  2. Baseline schema: [model_name, model_version, bins, bin_min, bin_max, rate]
  3. Detail schema:   [oroj_no, pre_project_no, cust_id, cert_id, score, apply_date]
  4. Bin scores by [bin_min, bin_max) (left-closed, right-open)
  5. Compute overall PSI and per-quarter PSI
  6. Write results to Hive partitioned table (partition col = pt, table = TAB_NAME)

PSI = sum( (p_i - q_i) * ln(p_i / q_i) )
  q_i = expected rate (baseline), p_i = actual rate
"""

import numpy as np
import pandas as pd
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType, DoubleType, DateType

EPS = 1e-6  # smoothing constant to avoid div-by-zero / log(0)


# ============================================================
# Logging helpers
# ============================================================

def log_step(step_name):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] START: {step_name}")


def log_step_end(step_name, result_msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] END:   {step_name}")
    print(f"  result: {result_msg}")
    print()


# ============================================================
# 1. Init SparkSession
# ============================================================

def init_spark():
    """Init SparkSession with Hive support."""
    spark = (
        SparkSession.builder
        .appName("PSI_Calculator")
        .enableHiveSupport()
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# 2. Read data from Hive
# ============================================================

def read_detail(detail_sql):
    """
    Read detail via spark.sql().
    Expected fields: oroj_no, pre_project_no, cust_id, cert_id, score, apply_date
    Returns Spark DataFrame.
    """
    spark = SparkSession.getActiveSession()
    df = spark.sql(detail_sql)
    df = df.select(
        F.col("oroj_no").cast(StringType()).alias("oroj_no"),
        F.col("pre_project_no").cast(StringType()).alias("pre_project_no"),
        F.col("cust_id").cast(StringType()).alias("cust_id"),
        F.col("cert_id").cast(StringType()).alias("cert_id"),
        F.col("score").cast(IntegerType()).alias("score"),
        F.col("apply_date").cast(DateType()).alias("apply_date"),
    )
    return df


def read_baseline(baseline_sql):
    """
    Read baseline via spark.sql().
    Expected fields: model_name, model_version, bins, bin_min, bin_max, rate
    Returns Spark DataFrame.
    """
    spark = SparkSession.getActiveSession()
    df = spark.sql(baseline_sql)
    df = df.select(
        F.col("model_name").cast(StringType()).alias("model_name"),
        F.col("model_version").cast(StringType()).alias("model_version"),
        F.col("bins").cast(StringType()).alias("bins"),
        F.col("bin_min").cast(IntegerType()).alias("bin_min"),
        F.col("bin_max").cast(IntegerType()).alias("bin_max"),
        F.col("rate").cast(DoubleType()).alias("rate"),
    )
    return df


# ============================================================
# 3. Add quarter column
# ============================================================

def add_quarter(df_detail):
    """
    Add 'quarter' column to df_detail, format 'YYYY-QN'.
    Based on apply_date.
    """
    df = df_detail.withColumn(
        "quarter",
        F.concat(
            F.year(F.col("apply_date")).cast("string"),
            F.lit("-Q"),
            F.floor((F.month(F.col("apply_date")) - 1) / 3 + 1).cast("string"),
        ),
    )
    return df


# ============================================================
# 4. Binning & PSI calculation
# ============================================================

def assign_to_bins(df_detail, df_baseline, quarter_col=None):
    """
    Bin df_detail scores by df_baseline bin_min/bin_max (left-closed, right-open).

    Params:
      df_detail:   Spark DataFrame with 'score' (and optional 'quarter')
      df_baseline:  Spark DataFrame with model_name, model_version, bins,
                    bin_min, bin_max, rate
      quarter_col:  if not None, also group by quarter

    Returns: pandas DataFrame
      overall:  [model_name, model_version, bins, bin_min, bin_max, rate, actual_rate]
      quarter:  [model_name, model_version, quarter, bins, bin_min, bin_max,
                 rate, actual_rate, sample_cnt]
    """
    spark = SparkSession.getActiveSession()

    bins_data = df_baseline.select(
        "model_name", "model_version", "bins", "bin_min", "bin_max", "rate"
    ).collect()
    n_bins = len(bins_data)

    model_name = bins_data[0]["model_name"] if n_bins > 0 else ""
    model_version = bins_data[0]["model_version"] if n_bins > 0 else ""

    total = df_detail.count()

    if quarter_col is not None:
        # ---- Quarter mode ----
        conditions = []
        for b in bins_data:
            cond = f"(score >= {b['bin_min']} AND score < {b['bin_max']})"
            conditions.append(cond)

        bin_expr = "CASE " + " ".join(
            [f"WHEN {conditions[i]} THEN {i}" for i in range(n_bins)]
        ) + " ELSE -1 END"

        df_binned = df_detail.select(
            F.col(quarter_col).alias("quarter"),
            F.expr(bin_expr).alias("bin_id"),
            F.lit(1).alias("cnt"),
        )

        # Per-quarter total (for actual_rate calculation)
        df_q_total = (
            df_binned.groupBy("quarter")
            .agg(F.sum("cnt").alias("q_total"))
        )

        # Per-quarter per-bin count
        df_counts = (
            df_binned.groupBy("quarter", "bin_id")
            .agg(F.sum("cnt").alias("cnt"))
        )

        # Cross join quarters x bins to ensure all combinations exist
        quarters_df = df_binned.select("quarter").distinct()
        bin_ids_df = spark.createDataFrame(
            [(i,) for i in range(n_bins)],
            schema=["bin_id"],
        )
        cross = quarters_df.crossJoin(bin_ids_df)

        # Left join actual counts (missing combos get 0)
        df_counts = cross.join(df_counts, on=["quarter", "bin_id"], how="left")
        df_counts = df_counts.fillna({"cnt": 0})

        # Join q_total
        df_counts = df_counts.join(df_q_total, on="quarter", how="left")
        df_counts = df_counts.withColumn("actual_rate", F.col("cnt") / F.col("q_total"))

        # Join baseline bin info
        bins_df = spark.createDataFrame(
            [(i, b["bins"], int(b["bin_min"]), int(b["bin_max"]), float(b["rate"]))
             for i, b in enumerate(bins_data)],
            schema=["bin_id", "bins", "bin_min", "bin_max", "rate"],
        )

        df_counts = df_counts.join(bins_df, on="bin_id", how="left")

        result = df_counts.select(
            F.lit(model_name).alias("model_name"),
            F.lit(model_version).alias("model_version"),
            "quarter",
            "bins",
            "bin_min",
            "bin_max",
            "rate",
            "actual_rate",
            F.col("q_total").alias("sample_cnt"),
        ).orderBy("quarter", "bin_id")

        pdf = result.toPandas()
        pdf["rate"] = pdf["rate"].astype(float)
        pdf["actual_rate"] = pdf["actual_rate"].astype(float)
        pdf["sample_cnt"] = pdf["sample_cnt"].astype(int)
        return pdf

    else:
        # ---- Overall mode ----
        counts_list = []
        for b in bins_data:
            cnt = df_detail.filter(
                (F.col("score") >= b["bin_min"]) & (F.col("score") < b["bin_max"])
            ).count()
            counts_list.append({
                "model_name": model_name,
                "model_version": model_version,
                "bins": b["bins"],
                "bin_min": int(b["bin_min"]),
                "bin_max": int(b["bin_max"]),
                "rate": float(b["rate"]),
                "actual_rate": cnt / total if total > 0 else 0.0,
            })

        pdf = pd.DataFrame(counts_list)
        return pdf


def calculate_psi_row(row):
    """
    Single-row PSI: psi_i = (p - q) * ln(p / q), with smoothing.
    q = rate (expected), p = actual_rate
    """
    q = row["rate"] if row["rate"] > EPS else EPS
    p = row["actual_rate"] if row["actual_rate"] > EPS else EPS
    return (p - q) * np.log(p / q)


def calculate_psi(df_actual):
    """
    Input:  pandas DataFrame with 'rate' and 'actual_rate' columns
    Output: (psi_total, df_result)
      - psi_total: float, sum of all bin psi values
      - df_result: DataFrame with 'psi' column added
    """
    df_result = df_actual.copy()
    df_result["psi"] = df_result.apply(calculate_psi_row, axis=1)
    psi_total = df_result["psi"].sum()
    return psi_total, df_result


# ============================================================
# 5. Write to Hive (partitioned table)
# ============================================================

def write_to_hive(pdf, tab_name, pt, mode="overwrite"):
    """
    Write pandas DataFrame to Hive partitioned table.
    Partition column = pt.

    Params:
      pdf:     pandas DataFrame
      tab_name: target Hive table name
      pt:      partition value (string)
      mode:    write mode
    """
    spark = SparkSession.getActiveSession()
    sdf = spark.createDataFrame(pdf)
    (
        sdf.write
        .mode(mode)
        .partitionBy("pt")
        .saveAsTable(tab_name)
    )


# ============================================================
# 6. Main pipeline (try/except)
# ============================================================

def main(tab_name, pt, detail_sql, baseline_sql):
    """
    Main entry point.

    Params:
      tab_name:      target Hive table name for PSI results
      pt:            partition value (e.g. '2026-10-22')
      detail_sql:    Hive SQL to read detail table
      baseline_sql:  Hive SQL to read baseline table
    """
    # ----------------------------------------------------------
    # Step 1: Init SparkSession
    # ----------------------------------------------------------
    step = "Step 1: init SparkSession"
    log_step(step)
    try:
        spark = init_spark()
        log_step_end(step, "SparkSession initialized")
    except Exception as e:
        log_step_end(step, f"failed: {e}")
        return None

    # ----------------------------------------------------------
    # Step 2: Read detail from Hive
    # ----------------------------------------------------------
    step = "Step 2: read detail from Hive"
    log_step(step)
    try:
        df_detail = read_detail(detail_sql)
        total_count = df_detail.count()
        cols = df_detail.columns
        log_step_end(step, f"{total_count} rows, columns: {cols}")
    except Exception as e:
        log_step_end(step, f"failed: {e}")
        spark.stop()
        return None

    # ----------------------------------------------------------
    # Step 3: Read baseline from Hive
    # ----------------------------------------------------------
    step = "Step 3: read baseline from Hive"
    log_step(step)
    try:
        df_baseline = read_baseline(baseline_sql)
        baseline_count = df_baseline.count()
        cols = df_baseline.columns
        log_step_end(step, f"{baseline_count} bins, columns: {cols}")
    except Exception as e:
        log_step_end(step, f"failed: {e}")
        spark.stop()
        return None

    # ----------------------------------------------------------
    # Step 4: Compute overall PSI
    # ----------------------------------------------------------
    step = "Step 4: compute overall PSI"
    log_step(step)
    try:
        pdf_overall = assign_to_bins(df_detail, df_baseline, quarter_col=None)
        psi_overall, pdf_overall = calculate_psi(pdf_overall)
        log_step_end(step, f"overall PSI = {psi_overall:.6f}")
    except Exception as e:
        log_step_end(step, f"failed: {e}")
        spark.stop()
        return None

    # ----------------------------------------------------------
    # Step 5: Compute per-quarter PSI
    # ----------------------------------------------------------
    step = "Step 5: compute per-quarter PSI"
    log_step(step)
    try:
        df_detail_q = add_quarter(df_detail)

        pdf_quarter_binned = assign_to_bins(
            df_detail_q, df_baseline, quarter_col="quarter"
        )

        # Ensure clean RangeIndex right after toPandas
        pdf_quarter_binned = pdf_quarter_binned.reset_index(drop=True)

        # Drop rows where bins is None (score outside all bin ranges, bin_id=-1)
        # and drop duplicate rows caused by Spark joins
        pdf_quarter_binned = pdf_quarter_binned.dropna(subset=["bins", "rate", "actual_rate"])
        pdf_quarter_binned = pdf_quarter_binned.drop_duplicates(
            subset=["quarter", "bins"]
        )
        pdf_quarter_binned = pdf_quarter_binned.reset_index(drop=True)

        # Compute psi per row (vectorized formula instead of apply)
        q_vals = pdf_quarter_binned["rate"].clip(lower=EPS).values
        p_vals = pdf_quarter_binned["actual_rate"].clip(lower=EPS).values
        psi_vals = (p_vals - q_vals) * np.log(p_vals / q_vals)

        # Build detail DataFrame with target Hive table column names:
        # [model_name, model_version, obs_date, bins, base_line, acutal_line, psi_bin, pt]
        n = len(pdf_quarter_binned)
        df_quarter_psi = pd.DataFrame({
            "model_name": pdf_quarter_binned["model_name"].values,
            "model_version": pdf_quarter_binned["model_version"].values,
            "obs_date": pdf_quarter_binned["quarter"].values,
            "bins": pdf_quarter_binned["bins"].values,
            "base_line": np.round(pdf_quarter_binned["rate"].values, 4),
            "acutal_line": np.round(pdf_quarter_binned["actual_rate"].values, 4),
            "psi_bin": np.round(psi_vals, 6),
            "pt": np.array([pt] * n),
        })

        log_step_end(
            step,
            f"{df_quarter_psi.shape[0]} rows x {df_quarter_psi.shape[1]} cols, "
            f"columns: {list(df_quarter_psi.columns)}"
        )
    except Exception as e:
        log_step_end(step, f"failed: {e}")
        spark.stop()
        return None

    # ----------------------------------------------------------
    # Step 6: Write PSI result to Hive (partitioned)
    # ----------------------------------------------------------
    step = "Step 6: write PSI result to Hive"
    log_step(step)
    try:
        write_to_hive(df_quarter_psi, tab_name, pt)
        log_step_end(
            step,
            f"written to {tab_name} (pt={pt}), "
            f"{df_quarter_psi.shape[0]} rows x {df_quarter_psi.shape[1]} cols"
        )
    except Exception as e:
        log_step_end(step, f"failed: {e}")

    # ----------------------------------------------------------
    # Step 7: Print PSI result
    # ----------------------------------------------------------
    step = "Step 7: print PSI result"
    log_step(step)
    try:
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", 140)

        result_str = df_quarter_psi.to_string(index=False)
        log_step_end(step, f"\n{result_str}")
    except Exception as e:
        log_step_end(step, f"failed: {e}")

    # ----------------------------------------------------------
    # Step 8: Stop SparkSession
    # ----------------------------------------------------------
    step = "Step 8: stop SparkSession"
    log_step(step)
    try:
        spark.stop()
        log_step_end(step, "SparkSession stopped")
    except Exception as e:
        log_step_end(step, f"failed: {e}")

    return df_quarter_psi


if __name__ == "__main__":
    # ---- Parameters ----
    TAB_NAME = "your_db.psi_result"
    PT = "2026-10-22"

    DETAIL_SQL = (
        "SELECT oroj_no, pre_project_no, cust_id, cert_id, score, apply_date "
        "FROM your_db.your_detail_table"
    )
    BASELINE_SQL = (
        "SELECT model_name, model_version, bins, bin_min, bin_max, rate "
        "FROM your_db.your_baseline_table"
    )

    main(
        tab_name=TAB_NAME,
        pt=PT,
        detail_sql=DETAIL_SQL,
        baseline_sql=BASELINE_SQL,
    )
