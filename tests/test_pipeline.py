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


def test_zero_assets_do_not_create_infinities():
    panel = synthetic_panel(n_companies=60)
    panel.loc[(panel["cik"] == 5) & (panel["fiscal_year"] == 2015), "assets"] = 0.0
    df = forecast.add_target_and_features(metrics.add_metrics(panel))
    cols = ["op_roa", "accruals", "asset_growth", "target_next_op_roa"]
    assert not np.isinf(df[cols].to_numpy(dtype=float)).any()


def test_split_balance_sheet_date_is_collapsed_into_one_row():
    # 52/53 week year: income statement ends 2020-12-31, balance sheet 2021-01-02.
    facts = {
        "cik": 7,
        "facts": {"us-gaap": {
            "Revenues": {"units": {"USD": [
                {"val": 100, "start": "2020-01-01", "end": "2020-12-31", "filed": "2021-02-15", "form": "10-K"},
            ]}},
            "Assets": {"units": {"USD": [
                {"val": 500, "end": "2021-01-02", "filed": "2021-02-15", "form": "10-K"},
            ]}},
        }},
    }
    panel = extract.annual_panel(facts)
    assert len(panel) == 1
    assert panel.loc[0, "revenue"] == 100 and panel.loc[0, "assets"] == 500


def test_operating_income_falls_back_to_revenue_minus_costs():
    panel = synthetic_panel(n_companies=3, years=range(2018, 2021))
    panel["costs_and_expenses"] = panel["revenue"] - panel["op_income"]
    expected = panel["op_income"].copy()
    panel["op_income"] = np.nan
    out = metrics.add_metrics(panel)
    np.testing.assert_allclose(out.sort_index()["op_income"].to_numpy(), expected.to_numpy())


def test_min_metrics_two_scores_companies_with_fewer_metrics():
    panel = synthetic_panel(n_companies=40, years=range(2015, 2019))
    panel["gross_profit"] = np.nan
    panel["cogs"] = np.nan      # no gross profit possible
    panel["op_income"] = np.nan  # no operating margin possible
    df = metrics.add_metrics(panel)
    assert metrics.rank_current(df, min_metrics=3)["current_score"].isna().all()
    assert metrics.rank_current(df, min_metrics=2)["current_score"].notna().any()


def test_equity_only_rows_are_dropped():
    # A later 10-K lists opening equity for 2013, before the company had real data.
    facts = fake_facts()
    facts["facts"]["us-gaap"]["StockholdersEquity"] = {"units": {"USD": [
        {"val": -120, "end": "2013-12-31", "filed": "2021-02-15", "form": "10-K"},
        {"val": 200, "end": "2020-12-31", "filed": "2021-02-15", "form": "10-K"},
    ]}}
    panel = extract.annual_panel(facts)
    assert panel["fiscal_year"].tolist() == [2020]
    assert panel.loc[0, "equity"] == 200
