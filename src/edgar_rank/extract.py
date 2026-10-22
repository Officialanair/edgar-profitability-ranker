"""Turn a companyfacts JSON blob into an annual, as-originally-reported panel."""
import pandas as pd

from . import config


def _rows_for_tag(facts_usgaap: dict, tag: str, flow: bool):
    node = facts_usgaap.get(tag)
    if not node:
        return []
    out = []
    for e in node.get("units", {}).get("USD", []):
        if e.get("form") not in config.ANNUAL_FORMS:
            continue
        end = pd.to_datetime(e.get("end"), errors="coerce")
        filed = pd.to_datetime(e.get("filed"), errors="coerce")
        if pd.isna(end) or pd.isna(filed):
            continue
        if flow:
            start = pd.to_datetime(e.get("start"), errors="coerce")
            if pd.isna(start):
                continue
            days = (end - start).days
            if not (config.MIN_DURATION_DAYS <= days <= config.MAX_DURATION_DAYS):
                continue
        out.append({"period_end": end, "filed": filed, "value": e.get("val")})
    return out


def _first_available(facts_usgaap: dict, tags: list, flow: bool) -> pd.DataFrame:
    """For each period_end, take the highest-priority tag that has data, using the
    earliest filing for that period (as originally reported, no restatements)."""
    frames = []
    for priority, tag in enumerate(tags):
        rows = _rows_for_tag(facts_usgaap, tag, flow)
        if rows:
            df = pd.DataFrame(rows)
            df["priority"] = priority
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["period_end", "filed", "value"])
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["period_end", "priority", "filed"])
    return df.drop_duplicates("period_end", keep="first")[["period_end", "filed", "value"]]


def annual_panel(facts_json: dict) -> pd.DataFrame:
    """One row per fiscal period with one column per concept plus avail_date,
    the latest filing date among the concepts (when the row was fully knowable)."""
    if not facts_json:
        return pd.DataFrame()
    usgaap = facts_json.get("facts", {}).get("us-gaap", {})
    pieces = {}
    for name, tags in config.FLOW_TAGS.items():
        pieces[name] = _first_available(usgaap, tags, flow=True)
    for name, tags in config.INSTANT_TAGS.items():
        pieces[name] = _first_available(usgaap, tags, flow=False)

    panel = None
    filed_cols = []
    for name, df in pieces.items():
        if df.empty:
            continue
        d = df.rename(columns={"value": name, "filed": f"filed_{name}"})
        filed_cols.append(f"filed_{name}")
        panel = d if panel is None else panel.merge(d, on="period_end", how="outer")
    if panel is None or panel.empty:
        return pd.DataFrame()

    panel["avail_date"] = panel[filed_cols].max(axis=1)
    panel = panel.drop(columns=filed_cols)
    # Label by fiscal year. Shifting back 31 days maps Jan/early-Feb year ends
    # (common for retailers) to the prior fiscal year.
    panel["fiscal_year"] = (panel["period_end"] - pd.Timedelta(days=31)).dt.year
    panel["cik"] = int(facts_json.get("cik", 0))
    # Income-statement and balance-sheet dates can differ by a day or two (52/53
    # week years), which splits one fiscal year into partial rows. Collapse rows
    # that share a fiscal year, taking the last non-null value in each column.
    panel = panel.sort_values("period_end")
    agg = {c: "last" for c in panel.columns if c != "fiscal_year"}
    agg["period_end"] = "max"
    agg["avail_date"] = "max"
    return panel.groupby("fiscal_year", as_index=False).agg(agg)
