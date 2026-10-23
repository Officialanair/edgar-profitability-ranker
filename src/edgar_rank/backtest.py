"""Backtest: how well does each predictor rank next year's profitability?

Every predictor is scored on the same rows each year (the "common sample"), so a
model is not rewarded just for covering easier companies. For each fiscal year Y
we compare the predictor's value at Y with the company's realized percentile rank
of operating ROA in Y + 1.

Metrics per year:
  spearman     rank correlation between predictor and realized next-year rank
  top_persist  share of the predictor's top quintile that is in the realized top
               quintile next year (a random pick would score about 0.20)
  top_mean_rank average realized next-year rank of that top quintile (random is 0.50)
"""
import numpy as np
import pandas as pd

PREDICTORS = {
    "naive_last_year": "op_roa_rank",   # "this year's rank repeats"
    "current_score": "current_score",   # composite of four current metrics
    "shrinkage": "shrink_pred",
    "regression": "reg_pred",
    "projected_blend": "projected_score",
    "final_score": "final_score",
}
TARGET = "target_next_rank"


def evaluate(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    cols = list(PREDICTORS.values()) + [TARGET]
    rows = []
    for year, g in df.groupby("fiscal_year"):
        common = g.dropna(subset=cols)
        if len(common) < min_n:
            continue
        for name, col in PREDICTORS.items():
            x, y = common[col], common[TARGET]
            top = x >= x.quantile(0.8)
            rows.append({
                "fiscal_year": int(year), "predictor": name, "n": len(common),
                "spearman": x.corr(y, method="spearman"),
                "top_persist": float((y[top] >= 0.8).mean()),
                "top_mean_rank": float(y[top].mean()),
            })
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    """Average each predictor across years and compare it with the naive baseline."""
    if bt.empty:
        return bt
    wide = bt.pivot(index="fiscal_year", columns="predictor", values="spearman")
    naive = wide["naive_last_year"]
    g = bt.groupby("predictor")
    out = pd.DataFrame({
        "years": g["spearman"].count(),
        "mean_spearman": g["spearman"].mean(),
        "std_error": g["spearman"].std() / np.sqrt(g["spearman"].count()),
        "mean_top_persist": g["top_persist"].mean(),
        "mean_top_rank": g["top_mean_rank"].mean(),
    })
    out["years_beating_naive"] = [
        int((wide[p] > naive).sum()) if p != "naive_last_year" else np.nan for p in out.index
    ]
    order = [p for p in PREDICTORS if p in out.index]
    return out.loc[order].round(3)
