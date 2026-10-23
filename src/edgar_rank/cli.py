"""Command line entry points.

    python -m edgar_rank.cli build --limit 200      # download + build the panel
    python -m edgar_rank.cli rank --alpha 0.5       # score and rank each fiscal year
    python -m edgar_rank.cli backtest               # does the ranking predict next-year profitability?
"""
import argparse
import random

import numpy as np
import pandas as pd

from . import backtest, config, edgar, extract, forecast, metrics


def build(limit: int | None, cik_file: str | None, seed: int = 0) -> None:
    session = edgar._session()
    if cik_file:
        ciks = [int(x) for x in open(cik_file).read().split()]
    else:
        # Note: this lists CURRENT registrants only, which biases the sample toward
        # survivors. Pass --cik-file with historical CIKs to fix that.
        ciks = sorted({v["cik_str"] for v in edgar.company_tickers(session).values()})
    if limit:
        # Random sample, not the first N. Low CIKs are the oldest registrants, so
        # taking the first N would skew the sample toward legacy companies.
        ciks = sorted(random.Random(seed).sample(ciks, min(limit, len(ciks))))

    frames = []
    for i, cik in enumerate(ciks, 1):
        try:
            sub = edgar.submissions(cik, session)
            if not sub or edgar.sic_is_excluded(sub.get("sic")):
                continue
            panel = extract.annual_panel(edgar.company_facts(cik, session))
            if panel.empty:
                continue
            panel["sic"] = pd.to_numeric(sub.get("sic"), errors="coerce")
            panel["name"] = sub.get("name")
            frames.append(panel)
        except Exception as exc:  # keep going; one bad company should not stop the run
            print(f"skip CIK {cik}: {exc}")
        if i % 50 == 0:
            print(f"{i}/{len(ciks)} companies processed")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(config.OUT_DIR / "panel.csv", index=False)
    print(f"wrote {len(out)} company-years for {out['cik'].nunique()} companies")


def usable_years(df: pd.DataFrame, min_year: int, min_share: float = 0.5) -> list:
    """Fiscal years to score: from min_year on, and with at least min_share of the
    median year's company count. This drops XBRL-phase-in years and a partial
    latest year, whose cross-section is not comparable."""
    counts = df[df["fiscal_year"] >= min_year].groupby("fiscal_year")["cik"].nunique()
    return counts[counts >= min_share * counts.median()].index.tolist()


def score_panel(panel: pd.DataFrame, alpha: float = 0.5, min_year: int = 2010,
                min_metrics: int = 3, verbose: bool = True) -> pd.DataFrame:
    """Full scoring pipeline on a panel. Returns one row per scored company-year with
    current, shrinkage, regression, blended and final scores plus the realized
    next-year rank (used by the backtest)."""
    panel = panel.copy()
    panel["sic"] = panel["sic"].astype("Int64")
    panel["sic2"] = panel["sic"] // 100
    df = metrics.add_metrics(panel)
    years = usable_years(df, min_year)
    if verbose:
        dropped = sorted(set(df["fiscal_year"].unique()) - set(years))
        print(f"scoring fiscal years {min(years)} to {max(years)}; skipped {dropped}")

    # Targets only point at scored years, so a partial latest year is never an outcome.
    df = forecast.add_target_and_features(df, valid_years=years)
    # The shrinkage baseline uses trailing history, so compute it before dropping early rows.
    df["shrink_pred"] = forecast.shrinkage_forecast(df)
    df = df[df["fiscal_year"].isin(years)].copy()
    df = metrics.rank_current(df, min_metrics=min_metrics)

    if verbose:
        unscored = df["current_score"].isna()
        print(f"unscored company-years: {unscored.mean():.0%}. Missing share by field (scored vs unscored):")
        fields = ["revenue", "gp_filled", "op_income", "net_income", "cfo", "assets"]
        print(df.groupby(unscored)[fields].apply(lambda d: d.isna().mean().round(2)).T.rename(
            columns={False: "scored", True: "unscored"}).to_string())

    # Regression predictions where enough earlier years exist; shrinkage covers the rest.
    df["reg_pred"] = np.nan
    for year in sorted(df["fiscal_year"].unique()):
        pred = forecast.regression_forecast(df, int(year))
        df.loc[pred.index, "reg_pred"] = pred
    df["projected_score"] = df["reg_pred"].fillna(df["shrink_pred"])
    df["projected_op_roa"] = forecast.rank_to_level(df, "projected_score")
    return forecast.blend_scores(df, alpha=alpha)


def _load_panel() -> pd.DataFrame:
    return pd.read_csv(config.OUT_DIR / "panel.csv", parse_dates=["period_end", "avail_date"])


def rank(alpha: float, min_year: int = 2010, min_metrics: int = 3) -> None:
    df = score_panel(_load_panel(), alpha, min_year, min_metrics)
    cols = ["fiscal_year", "final_rank", "cik", "name", "sic", "current_score",
            "current_score_industry", "projected_op_roa", "projected_rank", "final_score"]
    out = df[[c for c in cols if c in df]].sort_values(["fiscal_year", "final_rank"])
    out.to_csv(config.OUT_DIR / "rankings.csv", index=False)
    print(f"wrote rankings for fiscal years {int(out['fiscal_year'].min())} to {int(out['fiscal_year'].max())}")


def run_backtest(alpha: float, min_year: int = 2010, min_metrics: int = 3) -> None:
    df = score_panel(_load_panel(), alpha, min_year, min_metrics, verbose=False)
    bt = backtest.evaluate(df)
    if bt.empty:
        print("not enough overlapping company-years to backtest; build a larger panel")
        return
    bt.to_csv(config.OUT_DIR / "backtest.csv", index=False)
    summary = backtest.summarize(bt)
    print(f"Backtest over fiscal years {bt['fiscal_year'].min()} to {bt['fiscal_year'].max()} "
          f"({bt['fiscal_year'].nunique()} years, about {int(bt['n'].median())} companies per year)")
    print("Higher is better. A random pick scores top_persist 0.20 and top_rank 0.50.\n")
    print(summary.to_string())
    print("\nIf two predictors are within about two std_error of each other, treat them as tied.")
    print(f"wrote per-year results to {config.OUT_DIR / 'backtest.csv'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--limit", type=int)
    b.add_argument("--cik-file", help="text file of CIKs, whitespace separated")
    b.add_argument("--seed", type=int, default=0, help="seed for the random sample used with --limit")
    r = sub.add_parser("rank")
    r.add_argument("--alpha", type=float, default=0.5, help="weight on current vs projected")
    r.add_argument("--min-year", type=int, default=2010, help="first fiscal year to score")
    r.add_argument("--min-metrics", type=int, default=3,
                   help="how many of the 4 profitability metrics a company-year needs to be scored")
    t = sub.add_parser("backtest", help="compare predictors against realized next-year profitability")
    t.add_argument("--alpha", type=float, default=0.5)
    t.add_argument("--min-year", type=int, default=2010)
    t.add_argument("--min-metrics", type=int, default=3)
    args = p.parse_args()
    if args.cmd == "build":
        build(args.limit, args.cik_file, args.seed)
    elif args.cmd == "rank":
        rank(args.alpha, args.min_year, args.min_metrics)
    else:
        run_backtest(args.alpha, args.min_year, args.min_metrics)


if __name__ == "__main__":
    main()
