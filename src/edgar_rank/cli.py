"""Command line entry points.

    python -m edgar_rank.cli build --limit 200      # download + build the panel
    python -m edgar_rank.cli rank --alpha 0.5       # score and rank each fiscal year
"""
import argparse

import pandas as pd

from . import config, edgar, extract, forecast, metrics


def build(limit: int | None, cik_file: str | None) -> None:
    session = edgar._session()
    if cik_file:
        ciks = [int(x) for x in open(cik_file).read().split()]
    else:
        # Note: this lists CURRENT registrants only, which biases the sample toward
        # survivors. Pass --cik-file with historical CIKs to fix that.
        ciks = sorted({v["cik_str"] for v in edgar.company_tickers(session).values()})
    if limit:
        ciks = ciks[:limit]

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


def rank(alpha: float) -> None:
    panel = pd.read_csv(config.OUT_DIR / "panel.csv", parse_dates=["period_end", "avail_date"])
    panel["sic2"] = (panel["sic"] // 100).astype("Int64")
    df = metrics.add_metrics(panel)
    df = forecast.add_target_and_features(df)
    df = metrics.rank_current(df)
    df["projected_op_roa"] = forecast.shrinkage_forecast(df)

    # Replace the shrinkage baseline with the regression where enough history exists.
    for year in sorted(df["fiscal_year"].unique()):
        pred = forecast.regression_forecast(df, int(year))
        df.loc[pred.index[pred.notna()], "projected_op_roa"] = pred.dropna()

    df = forecast.blend_scores(df, alpha=alpha)
    cols = ["fiscal_year", "final_rank", "cik", "name", "sic", "current_score",
            "current_score_industry", "projected_op_roa", "projected_rank", "final_score"]
    out = df[[c for c in cols if c in df]].sort_values(["fiscal_year", "final_rank"])
    out.to_csv(config.OUT_DIR / "rankings.csv", index=False)
    print(f"wrote rankings for fiscal years {int(out['fiscal_year'].min())} to {int(out['fiscal_year'].max())}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--limit", type=int)
    b.add_argument("--cik-file", help="text file of CIKs, whitespace separated")
    r = sub.add_parser("rank")
    r.add_argument("--alpha", type=float, default=0.5, help="weight on current vs projected")
    args = p.parse_args()
    if args.cmd == "build":
        build(args.limit, args.cik_file)
    else:
        rank(args.alpha)


if __name__ == "__main__":
    main()
