"""
Feature pipeline for Refinery Utilization project.

This module consolidates helpers detected in your notebooks:
- df_info
- missing_values
- describe_data
- extract_date_parts
- yes_no_to_binary

…plus end-to-end functions to:
- load raw CSVs
- aggregate accidents monthly (with severity index)
- standardize month keys in monthly tables
- merge into a single modeling table
- build X, y, and a preprocessing ColumnTransformer
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from functools import reduce
from typing import Tuple, List, Dict, Any, Optional


# -----------------------------
# Helpers from notebooks (recreated)
# -----------------------------
def df_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    print(f"Data Information for {name}")
    df.info()
    print(f"Shape: {df.shape}\n")


def missing_values(df: pd.DataFrame, name: str = "DataFrame") -> None:
    print(f"Missing Values for {name}")
    print(df.isna().sum(), "\n")


def describe_data(datasets: Dict[str, pd.DataFrame]) -> None:
    from IPython.display import display

    for name, df in datasets.items():
        print("=" * 80)
        print(f"{name}")
        display(df.describe(include="all"))
        print()


def extract_date_parts(
    datasets: Dict[str, pd.DataFrame], date_columns: Dict[str, List[str]]
) -> Dict[str, pd.DataFrame]:
    """Add Day/MonthNumber/MonthAbbr/Year for specified datetime columns, in-place style."""
    for name, df in datasets.items():
        if name in date_columns:
            for col in date_columns[name]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    df[f"{col}_Day"] = df[col].dt.day
                    df[f"{col}_Month_Number"] = df[col].dt.month
                    df[f"{col}_Month"] = df[col].dt.strftime("%b")
                    df[f"{col}_Year"] = df[col].dt.year
    return datasets


def yes_no_to_binary(series: pd.Series) -> pd.Series:
    m = {
        "Y": 1,
        "YES": 1,
        "TRUE": 1,
        "T": 1,
        "1": 1,
        "N": 0,
        "NO": 0,
        "FALSE": 0,
        "F": 0,
        "0": 0,
    }
    return series.astype(str).str.strip().str.upper().map(m)


# -----------------------------
# IO
# -----------------------------
def load_data(
    accidents_path: str,
    crude_rigs_path: str,
    refinery_capacity_path: str,
    refinery_inputs_path: str,
    refinery_utilization_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accidents = pd.read_csv(accidents_path)
    crude_rigs = pd.read_csv(crude_rigs_path)
    refinery_capacity = pd.read_csv(refinery_capacity_path)
    refinery_inputs = pd.read_csv(refinery_inputs_path)
    refinery_utilization = pd.read_csv(refinery_utilization_path)
    return (
        accidents,
        crude_rigs,
        refinery_capacity,
        refinery_inputs,
        refinery_utilization,
    )


# -----------------------------
# Month key builders
# -----------------------------
def _to_period_from_date(
    df: pd.DataFrame, date_col: str, out_col: str = "Date_Month"
) -> pd.DataFrame:
    g = df.copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
    g = g.dropna(subset=[date_col])
    g[out_col] = g[date_col].dt.to_period("M")
    return g


def _to_period_from_year_month(
    df: pd.DataFrame, year_col: str, month_num_col: str, out_col: str = "Date_Month"
) -> pd.DataFrame:
    g = df.copy()
    g["_dt"] = pd.to_datetime(
        dict(year=g[year_col].astype(int), month=g[month_num_col].astype(int), day=1),
        errors="coerce",
    )
    g = g.dropna(subset=["_dt"])
    g[out_col] = g["_dt"].dt.to_period("M")
    return g.drop(columns=["_dt"])


# -----------------------------
# Accidents monthly aggregation + Severity Index
# -----------------------------
def aggregate_accidents_monthly(accidents: pd.DataFrame) -> pd.DataFrame:
    g = accidents.copy()
    # date
    g["LOCAL_DATETIME"] = pd.to_datetime(g["LOCAL_DATETIME"], errors="coerce")
    g = g.dropna(subset=["LOCAL_DATETIME"])
    g["Date_Month"] = g["LOCAL_DATETIME"].dt.to_period("M")

    # indicators
    for col in ["FATALITY_IND", "INJURY_IND", "WATER_CONTAM_IND"]:
        if col in g.columns:
            g[col] = yes_no_to_binary(g[col])

    # numeric coercion
    for col in [
        "UNINTENTIONAL_RELEASE_BBLS",
        "RECOVERED_BBLS",
        "EST_COST_PROP_DAMAGE",
        "EST_COST_ENVIRONMENTAL",
    ]:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    monthly = g.groupby("Date_Month", as_index=False).agg(
        Total_Accidents=("LOCAL_DATETIME", "count"),
        Total_Release_BBLS=("UNINTENTIONAL_RELEASE_BBLS", "sum"),
        Total_Recovered_BBLS=("RECOVERED_BBLS", "sum"),
        Avg_Prop_Damage=("EST_COST_PROP_DAMAGE", "mean"),
        Avg_Env_Damage=("EST_COST_ENVIRONMENTAL", "mean"),
        Total_Fatalities=("FATALITY_IND", "sum"),
        Total_Injuries=("INJURY_IND", "sum"),
    )

    # fill missing numeric
    fill_cols = [
        "Total_Release_BBLS",
        "Total_Recovered_BBLS",
        "Avg_Prop_Damage",
        "Avg_Env_Damage",
        "Total_Fatalities",
        "Total_Injuries",
    ]
    for c in fill_cols:
        if c in monthly.columns:
            monthly[c] = monthly[c].fillna(0)

    # Severity Index (0–1)
    norm_cols = [
        "Avg_Prop_Damage",
        "Avg_Env_Damage",
        "Total_Release_BBLS",
        "Total_Injuries",
        "Total_Fatalities",
    ]
    for c in norm_cols:
        cmin, cmax = monthly[c].min(), monthly[c].max()
        monthly[f"{c}_norm"] = (
            0 if cmax == cmin else (monthly[c] - cmin) / (cmax - cmin)
        )

    monthly["Severity_Index"] = (
        0.40 * monthly["Avg_Prop_Damage_norm"]
        + 0.30 * monthly["Avg_Env_Damage_norm"]
        + 0.15 * monthly["Total_Release_BBLS_norm"]
        + 0.10 * monthly["Total_Injuries_norm"]
        + 0.05 * monthly["Total_Fatalities_norm"]
    )

    monthly = monthly.sort_values("Date_Month")
    monthly["Severity_Index_3M"] = (
        monthly["Severity_Index"].rolling(3, min_periods=1).mean()
    )

    return monthly


# -----------------------------
# Prep month key for monthly tables
# -----------------------------
def ensure_month_key(
    df: pd.DataFrame,
    *,
    date_col="Date",
    year_col="Date_Year",
    month_num_col="Date_Month_Number",
) -> pd.DataFrame:
    if "Date_Month" in df.columns and pd.api.types.is_period_dtype(df["Date_Month"]):
        return df.copy()
    out = df.copy()
    if date_col in out.columns:
        out = _to_period_from_date(out, date_col, out_col="Date_Month")
    elif year_col in out.columns and month_num_col in out.columns:
        out = _to_period_from_year_month(
            out, year_col, month_num_col, out_col="Date_Month"
        )
    else:
        raise ValueError(
            "Need either 'Date' or ('Date_Year' + 'Date_Month_Number') to build Date_Month."
        )
    return out


# -----------------------------
# Merge into modeling table
# -----------------------------
def merge_monthly_tables(
    monthly_accidents: pd.DataFrame,
    crude_rigs: pd.DataFrame,
    refinery_capacity: pd.DataFrame,
    refinery_inputs: pd.DataFrame,
    refinery_utilization: pd.DataFrame,
) -> pd.DataFrame:
    rigs = ensure_month_key(crude_rigs)[
        ["Date_Month", "Operational_Rigs"]
    ].drop_duplicates("Date_Month")
    cap = ensure_month_key(refinery_capacity)[
        ["Date_Month", "Capacity_BBL"]
    ].drop_duplicates("Date_Month")
    inp = ensure_month_key(refinery_inputs)[
        ["Date_Month", "Input_BBL"]
    ].drop_duplicates("Date_Month")
    utl = ensure_month_key(refinery_utilization)[
        ["Date_Month", "Utilization_Percent"]
    ].drop_duplicates("Date_Month")

    frames = [monthly_accidents, rigs, cap, inp, utl]
    merged = reduce(lambda l, r: pd.merge(l, r, on="Date_Month", how="left"), frames)
    merged = merged.sort_values("Date_Month").reset_index(drop=True)
    return merged


# -----------------------------
# Convenience one-call builder
# -----------------------------
def build_feature_table_from_csvs(
    accidents_csv: str,
    crude_rigs_csv: str,
    capacity_csv: str,
    inputs_csv: str,
    util_csv: str,
) -> pd.DataFrame:
    acc, rigs, cap, inp, utl = load_data(
        accidents_csv, crude_rigs_csv, capacity_csv, inputs_csv, util_csv
    )
    monthly_acc = aggregate_accidents_monthly(acc)
    merged = merge_monthly_tables(monthly_acc, rigs, cap, inp, utl)
    return merged


if __name__ == "__main__":
    build_feature_table_from_csvs()
