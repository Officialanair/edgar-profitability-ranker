# EDGAR Profitability Ranker

Ranks US non-financial public companies each fiscal year by **current** and **projected** profitability, using only data from SEC EDGAR (XBRL company facts). Research tool, not investment advice.

## How it works

1. **Universe.** Pulls SIC codes from EDGAR submissions and drops SIC 6000 to 6799 (banks, insurers, REITs, funds).
2. **Panel.** Extracts annual (10-K) values per company from `companyfacts`, with tag fallbacks (`config.py`). Each value uses the earliest filing for that period, so you get numbers as originally reported, and each row carries an `avail_date` for look-ahead checks.
3. **Current score.** Percentile ranks within each fiscal year (after winsorizing) of gross profitability, ROA, operating margin, and FCF margin, averaged into a composite. A second score ranks within 2-digit SIC industry.
4. **Projected score.** Predicts each company's percentile rank of next-year operating income / assets, modeled in within-year rank space so a few huge losers (pre-revenue biotechs and shells) cannot skew the fit. A shrinkage baseline (trailing mean rank blended with the middle rank) is used first, then replaced by a ridge regression once enough earlier years exist. The regression trains only on years whose outcomes were already realized, and a partial latest year is never treated as an outcome. `projected_op_roa` in the output is the predicted rank translated back to a readable ratio using that year's actual distribution.
5. **Final rank.** `alpha * current + (1 - alpha) * projected`, default `alpha = 0.5`.

## Quick start

```bash
pip install -r requirements.txt
export EDGAR_USER_AGENT="Your Name your.email@example.com"   # required by the SEC
export PYTHONPATH=src

python -m edgar_rank.cli build --limit 200    # random sample (--seed N); drop --limit for everything
python -m edgar_rank.cli rank --alpha 0.5     # scores 2010 onward; skips partial years (--min-year to change)
python -m edgar_rank.cli backtest            # does the ranking predict next-year profitability?
pytest
```

Outputs land in `data/out/` (`panel.csv`, `rankings.csv`). Raw API responses are cached in `data/raw/`, so reruns are cheap. The client stays at 5 requests per second, under the SEC limit of 10.

## Known limitations (read before trusting results)

- **Survivorship bias.** The default universe comes from the current ticker list, so delisted companies are missing. Pass `--cik-file` with historical CIKs to fix this.
- **Tag coverage.** Revenue and operating income use many XBRL tags. Check coverage per year and extend `FLOW_TAGS` when a concept is missing.
- **As-originally-reported values** are used, but the SEC API does not expose every original filing for every fact. Spot check a few companies against their 10-Ks.
- **Utilities and other regulated industries** have structurally different margins. Prefer the industry score for those.
- The projected score is only as good as its backtest. Validate with year-by-year Spearman correlation of predicted vs realized profitability before relying on it.

## What the backtest says so far

On a random sample of about 90 filers (roughly 50 per year, 2012 to 2024), none of the projections beat the naive baseline "this year's profitability rank repeats". The regression and blended score are statistically tied with it, and the current-metrics composite is slightly behind, partly because the backtest target is operating ROA while the composite also uses gross profitability and cash flow. A sample this small cannot separate the models. Run the backtest on a much larger universe before drawing conclusions, and treat the projected half of the score as unproven until it clearly beats the naive baseline.

## Roadmap

- [x] Backtest report (`backtest` command: Spearman per year and top-quintile persistence vs a naive baseline)
- [ ] Gradient boosting forecast on the same features
- [ ] Coverage report for XBRL tag mapping
- [ ] Optional price data to test whether top-ranked firms earn better returns
