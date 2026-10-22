import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edgar_rank import extract, forecast, metrics  # noqa: E402


def fake_facts(cik=1):
    """Minimal companyfacts blob with an original value and a later restatement."""
    def flow(val, end, filed, start):
        return {"val": val, "start": start, "end": end, "filed": filed, "form": "10-K"}

    return {
        "cik": cik,
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                flow(100, "2020-12-31", "2021-02-15", "2020-01-01"),
                flow(90, "2020-12-31", "2022-02-15", "2020-01-01"),  # restated later
                flow(60, "2020-06-30", "2020-08-01", "2020-04-01"),  # quarter, must be dropped
            ]}},
            "Assets": {"units": {"USD": [
                {"val": 500, "end": "2020-12-31", "filed": "2021-02-15", "form": "10-K"},
            ]}},
        }},
    }


def test_extract_uses_original_value_and_drops_quarters():
    panel = extract.annual_panel(fake_facts())
    assert len(panel) == 1
    assert panel.loc[0, "revenue"] == 100
    assert panel.loc[0, "fiscal_year"] == 2020
    assert panel.loc[0, "avail_date"] == pd.Timestamp("2021-02-15")


def synthetic_panel(n_companies=120, years=range(2012, 2022), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cik in range(1, n_companies + 1):
        skill = rng.normal(0.08, 0.05)
        assets = rng.uniform(100, 1000)
        for fy in years:
            rev = assets * rng.uniform(0.5, 1.5)
            op = assets * (skill + rng.normal(0, 0.02))
            rows.append(dict(
                cik=cik, fiscal_year=fy, sic=2000 + (cik % 4) * 100,
                revenue=rev, cogs=rev * 0.6, gross_profit=np.nan, op_income=op,
                net_income=op * 0.7, cfo=op * 0.9, capex=assets * 0.03, assets=assets,
            ))
            assets *= 1.05
    p = pd.DataFrame(rows)
    p["sic2"] = p["sic"] // 100
    return p


def test_ranking_and_forecast_run_end_to_end():
    df = metrics.add_metrics(synthetic_panel())
    df = forecast.add_target_and_features(df)
    df = metrics.rank_current(df)
    assert df["current_score"].between(0, 1).all()
    assert df["current_score_industry"].notna().any()

    df["projected_op_roa"] = forecast.shrinkage_forecast(df)
    pred = forecast.regression_forecast(df, 2018)
    assert pred.notna().all()
    df.loc[pred.index, "projected_op_roa"] = pred

    out = forecast.blend_scores(df, alpha=0.5)
    one_year = out[out["fiscal_year"] == 2018]
    assert one_year["final_rank"].min() == 1
    assert one_year["final_rank"].is_unique


def test_regression_never_trains_on_future_years():
    df = forecast.add_target_and_features(metrics.add_metrics(synthetic_panel()))
    # Corrupt every target from the predicted year onward; predictions must not change.
    base = forecast.regression_forecast(df, 2016)
    poisoned = df.copy()
    poisoned.loc[poisoned["fiscal_year"] >= 2016, "target_next_op_roa"] = 99.0
    after = forecast.regression_forecast(poisoned, 2016)
    pd.testing.assert_series_equal(base, after)


def test_include_assessed_tax_revenue_is_picked_up():
    facts = fake_facts()
    facts["facts"]["us-gaap"] = {
        "RevenueFromContractWithCustomerIncludingAssessedTax": {"units": {"USD": [
            {"val": 250, "start": "2023-01-01", "end": "2023-12-31", "filed": "2024-02-20", "form": "10-K"},
        ]}},
    }
    panel = extract.annual_panel(facts)
    assert panel.loc[0, "revenue"] == 250


def test_usable_years_drops_early_and_partial_years():
    from edgar_rank import cli
    df = metrics.add_metrics(synthetic_panel(n_companies=100, years=range(2008, 2023)))
    # make 2022 partial: keep only 10 companies
    df = df[~((df["fiscal_year"] == 2022) & (df["cik"] > 10))]
    years = cli.usable_years(df, min_year=2010)
    assert 2009 not in years and 2022 not in years and 2015 in years
