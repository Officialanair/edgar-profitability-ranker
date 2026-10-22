"""Current-profitability metrics and cross-sectional ranking."""
import numpy as np
import pandas as pd

METRICS = ["gross_profitability", "roa", "op_margin", "fcf_margin"]


def add_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    """panel needs: cik, fiscal_year, revenue, gross_profit/cogs, op_income,
    net_income, cfo, capex, assets. Returns the panel with metric columns."""
    df = panel.sort_values(["cik", "fiscal_year"]).copy()
    df["assets_lag"] = df.groupby("cik")["assets"].shift(1)
    # Only trust the lag if it is truly the previous fiscal year.
    prev_fy = df.groupby("cik")["fiscal_year"].shift(1)
    df.loc[df["fiscal_year"] - prev_fy != 1, "assets_lag"] = np.nan
    df["avg_assets"] = df[["assets", "assets_lag"]].mean(axis=1)

    gp = df["gross_profit"]
    if "cogs" in df:
        gp = gp.fillna(df["revenue"] - df["cogs"])
    df["gp_filled"] = gp

    bad_assets = ~(df["avg_assets"] > 0)
    rev_ok = df["revenue"] > 0

    df["gross_profitability"] = (df["gp_filled"] / df["avg_assets"]).where(~bad_assets)
    df["roa"] = (df["net_income"] / df["avg_assets"]).where(~bad_assets)
    df["op_margin"] = (df["op_income"] / df["revenue"]).where(rev_ok)
    capex = df["capex"].fillna(0)
    df["fcf_margin"] = ((df["cfo"] - capex) / df["revenue"]).where(rev_ok)
    return df


def winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    if s.notna().sum() < 5:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def composite_score(df: pd.DataFrame, group: str | None = None, min_metrics: int = 3) -> pd.Series:
    """Average of percentile ranks of each metric within the fiscal year (and
    within `group`, such as a 2-digit SIC code, when given). Rows with fewer
    than `min_metrics` available metrics get NaN."""
    keys = ["fiscal_year"] + ([group] if group else [])
    ranked = pd.DataFrame(index=df.index)
    for m in METRICS:
        w = df.groupby(keys)[m].transform(winsorize)
        ranked[m] = w.groupby([df[k] for k in keys]).transform(_pct_rank)
    score = ranked.mean(axis=1)
    score[ranked.notna().sum(axis=1) < min_metrics] = np.nan
    return score


def rank_current(df: pd.DataFrame, min_group_size: int = 10) -> pd.DataFrame:
    """Adds current_score (whole universe) and current_score_industry (within
    2-digit SIC; falls back to the universe score for small industries)."""
    out = df.copy()
    out["current_score"] = composite_score(out)
    if "sic2" in out:
        ind = composite_score(out, group="sic2")
        size = out.groupby(["fiscal_year", "sic2"])["cik"].transform("count")
        out["current_score_industry"] = ind.where(size >= min_group_size, out["current_score"])
    return out
