"""Projected profitability, modeled in within-year percentile-rank space.

Why ranks: a random sample of US filers includes many pre-revenue companies
whose operating loss is several times their assets. Raw ratios let those few
rows dominate any average or regression fit and push every prediction negative.
Percentile ranks within a fiscal year are immune to that, and the final output
is a ranking anyway.

Both models predict a company's percentile rank of operating income / average
assets in the NEXT fiscal year. Neither uses information that was not available
at the fiscal year being scored, and the regression is trained only on years
whose outcomes were already realized.
"""
import numpy as np
import pandas as pd

# Raw features. Each gets a within-year percentile rank column named r_<feature>.
FEATURES = ["op_roa", "roa", "gross_profitability", "op_margin", "fcf_margin", "accruals", "asset_growth"]


def _year_rank(d: pd.DataFrame, col: str) -> pd.Series:
    return d.groupby("fiscal_year")[col].rank(pct=True)


def add_target_and_features(df: pd.DataFrame, valid_years=None) -> pd.DataFrame:
    """Adds ratio features, their within-year ranks, and the next-year target.

    valid_years: fiscal years allowed as a target year. Pass the list of years
    being scored so a partial latest year never becomes a "realized" outcome."""
    d = df.sort_values(["cik", "fiscal_year"]).copy()
    # Guard denominators: zero or negative asset bases would create infinities.
    avg_assets = d["avg_assets"].where(d["avg_assets"] > 0)
    d["op_roa"] = d["op_income"] / avg_assets
    d["accruals"] = (d["net_income"] - d["cfo"]) / avg_assets
    d["asset_growth"] = d["assets"] / d["assets_lag"].where(d["assets_lag"] > 0) - 1
    ratio_cols = ["op_roa", "accruals", "asset_growth"]
    d[ratio_cols] = d[ratio_cols].replace([np.inf, -np.inf], np.nan)

    for f in FEATURES:
        d[f"r_{f}"] = _year_rank(d, f)
    d["op_roa_rank"] = d["r_op_roa"]

    nxt_fy = d.groupby("cik")["fiscal_year"].shift(-1)
    consecutive = nxt_fy - d["fiscal_year"] == 1
    if valid_years is not None:
        consecutive &= (d["fiscal_year"] + 1).isin(set(valid_years))
    d["target_next_rank"] = d.groupby("cik")["op_roa_rank"].shift(-1).where(consecutive)
    d["target_next_op_roa"] = d.groupby("cik")["op_roa"].shift(-1).where(consecutive)
    return d


def shrinkage_forecast(df: pd.DataFrame, window: int = 3, weight: float = 0.6) -> pd.Series:
    """Blend each company's trailing mean rank with the middle rank (0.5).
    Profitability mean-reverts, so weight < 1 usually beats 'last year repeats'."""
    d = df.sort_values(["cik", "fiscal_year"])
    trailing = d.groupby("cik")["op_roa_rank"].transform(lambda s: s.rolling(window, min_periods=1).mean())
    return (weight * trailing + (1 - weight) * 0.5).reindex(df.index)


def _design(d: pd.DataFrame, features) -> np.ndarray:
    cols = [f"r_{f}" for f in features]
    x = d[cols].fillna(0.5) - 0.5  # missing feature -> neutral, centered ranks
    return np.column_stack([np.ones(len(x)), x.to_numpy()])


def regression_forecast(df: pd.DataFrame, predict_year: int, features=FEATURES, ridge: float = 5.0) -> pd.Series:
    """Ridge regression of next-year rank on this-year feature ranks.

    Trains only on fiscal years Y with Y + 1 <= predict_year, because the target
    for year Y is not realized until year Y + 1. Returns predicted ranks in [0, 1]
    for companies in `predict_year`, or NaN when there is too little history."""
    test = df[df["fiscal_year"] == predict_year]
    train = df[(df["fiscal_year"] + 1 <= predict_year) & df["target_next_rank"].notna()]
    if len(train) < 50 or test.empty:
        return pd.Series(np.nan, index=test.index)
    X = _design(train, features)
    y = train["target_next_rank"].to_numpy()
    penalty = ridge * np.eye(X.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    return pd.Series(np.clip(_design(test, features) @ beta, 0.0, 1.0), index=test.index)


def rank_to_level(df: pd.DataFrame, score_col: str = "projected_score") -> pd.Series:
    """Translate a predicted rank into an operating ROA estimate by reading that
    quantile off the same year's actual distribution. Only for readability."""
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("fiscal_year"):
        vals = g["op_roa"].dropna()
        scores = g[score_col].dropna()
        if len(vals) < 5 or scores.empty:
            continue
        out.loc[scores.index] = np.quantile(vals, scores.clip(0, 1).to_numpy())
    return out


def blend_scores(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """final_score = alpha * current percentile + (1 - alpha) * projected percentile."""
    out = df.copy()
    out["projected_rank"] = out.groupby("fiscal_year")["projected_score"].rank(pct=True)
    out["final_score"] = alpha * out["current_score"] + (1 - alpha) * out["projected_rank"]
    out["final_rank"] = out.groupby("fiscal_year")["final_score"].rank(ascending=False, method="first")
    return out
