"""Projected profitability: a shrinkage baseline and a cross-sectional regression.

Both predict next-year operating income / average assets. Neither uses any
information that was not available at the fiscal year being scored, and the
regression is trained only on years strictly before the year it predicts.
"""
import numpy as np
import pandas as pd

FEATURES = ["roa", "gross_profitability", "op_margin", "fcf_margin", "accruals", "asset_growth"]


def add_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values(["cik", "fiscal_year"]).copy()
    d["op_roa"] = d["op_income"] / d["avg_assets"]
    d["accruals"] = (d["net_income"] - d["cfo"]) / d["avg_assets"]
    d["asset_growth"] = d["assets"] / d["assets_lag"] - 1
    nxt = d.groupby("cik")["op_roa"].shift(-1)
    nxt_fy = d.groupby("cik")["fiscal_year"].shift(-1)
    d["target_next_op_roa"] = nxt.where(nxt_fy - d["fiscal_year"] == 1)
    return d


def shrinkage_forecast(df: pd.DataFrame, window: int = 3, weight: float = 0.6) -> pd.Series:
    """Blend each company's trailing mean operating ROA with the same-year
    cross-sectional mean. Profitability mean-reverts, so weight < 1 usually wins."""
    d = df.sort_values(["cik", "fiscal_year"])
    trailing = d.groupby("cik")["op_roa"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    xs_mean = d.groupby("fiscal_year")["op_roa"].transform("mean")
    return (weight * trailing + (1 - weight) * xs_mean).reindex(df.index)


def _design(d: pd.DataFrame, cols, means=None, stds=None):
    x = d[cols].copy()
    for c in cols:
        x[c] = x[c].clip(x[c].quantile(0.01), x[c].quantile(0.99))
    if means is None:
        means, stds = x.mean(), x.std().replace(0, 1)
    z = ((x - means) / stds).fillna(0.0)
    return np.column_stack([np.ones(len(z)), z.to_numpy()]), means, stds


def regression_forecast(df: pd.DataFrame, predict_year: int, features=FEATURES, ridge: float = 1.0) -> pd.Series:
    """Fit on fiscal years < predict_year (whose targets are fully realized by
    then), then predict next-year op ROA for companies in `predict_year`."""
    train = df[(df["fiscal_year"] < predict_year - 0) & df["target_next_op_roa"].notna()]
    # Target for fiscal_year Y is realized in Y+1, so only use Y+1 <= predict_year.
    train = train[train["fiscal_year"] + 1 <= predict_year]
    test = df[df["fiscal_year"] == predict_year]
    if len(train) < 50 or test.empty:
        return pd.Series(np.nan, index=test.index)
    X, means, stds = _design(train, features)
    y = train["target_next_op_roa"].clip(
        train["target_next_op_roa"].quantile(0.01), train["target_next_op_roa"].quantile(0.99)
    ).to_numpy()
    penalty = ridge * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    Xt, _, _ = _design(test, features, means, stds)
    return pd.Series(Xt @ beta, index=test.index)


def blend_scores(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """final_score = alpha * current percentile + (1 - alpha) * projected percentile."""
    out = df.copy()
    out["projected_rank"] = out.groupby("fiscal_year")["projected_op_roa"].rank(pct=True)
    out["final_score"] = alpha * out["current_score"] + (1 - alpha) * out["projected_rank"]
    out["final_rank"] = out.groupby("fiscal_year")["final_score"].rank(ascending=False, method="first")
    return out
