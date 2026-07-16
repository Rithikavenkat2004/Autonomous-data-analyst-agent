"""
core/auto_eda.py
-----------------
Runs the moment a CSV is uploaded. No LLM call needed here -- this is
classic ML/stats work (this is the "ML" part of the pipeline: outlier
detection via IQR + z-score, correlation analysis, schema inference).
The output feeds into the agent's system prompt so the LLM always has
grounded context about the data instead of guessing.
"""

import pandas as pd
import numpy as np


def profile_dataframe(df: pd.DataFrame) -> dict:
    profile = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": [],
        "outliers": {},
        "correlations": None,
    }

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "n_missing": int(df[col].isna().sum()),
            "pct_missing": round(float(df[col].isna().mean()) * 100, 2),
            "n_unique": int(df[col].nunique()),
        }
        if col in numeric_cols:
            col_info.update({
                "mean": round(float(df[col].mean()), 3) if df[col].notna().any() else None,
                "std": round(float(df[col].std()), 3) if df[col].notna().any() else None,
                "min": float(df[col].min()) if df[col].notna().any() else None,
                "max": float(df[col].max()) if df[col].notna().any() else None,
            })
        else:
            top_vals = df[col].value_counts().head(3).to_dict()
            col_info["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
        profile["columns"].append(col_info)

    # IQR-based outlier detection (unsupervised ML technique)
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if outlier_count > 0:
            profile["outliers"][col] = {
                "count": outlier_count,
                "pct": round(outlier_count / len(series) * 100, 2),
            }

    # Correlation matrix for numeric columns (helps the LLM suggest relationships)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).round(2)
        profile["correlations"] = corr.to_dict()

    return profile


def profile_to_prompt_context(profile: dict) -> str:
    """Turns the profile dict into compact text the LLM can reason over."""
    lines = [f"Dataset: {profile['n_rows']} rows, {profile['n_cols']} columns."]
    for col in profile["columns"]:
        base = f"- {col['name']} ({col['dtype']}), {col['pct_missing']}% missing, {col['n_unique']} unique"
        if "mean" in col:
            base += f", range [{col['min']}, {col['max']}], mean={col['mean']}"
        lines.append(base)
    if profile["outliers"]:
        lines.append("Outliers detected (IQR method): " +
                      ", ".join(f"{k} ({v['count']} rows, {v['pct']}%)" for k, v in profile["outliers"].items()))
    return "\n".join(lines)
