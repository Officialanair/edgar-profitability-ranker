"""Thin, polite client for the SEC EDGAR APIs with on-disk caching."""
import json
import time
from pathlib import Path

import requests

from . import config

_last_call = 0.0


def _session() -> requests.Session:
    if not config.USER_AGENT:
        raise RuntimeError(
            "Set EDGAR_USER_AGENT to something like 'Your Name you@example.com'. "
            "The SEC blocks requests without a descriptive User-Agent."
        )
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return s


def _throttle() -> None:
    global _last_call
    gap = 1.0 / config.MAX_REQUESTS_PER_SEC
    wait = gap - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _get_json(session: requests.Session, url: str, cache_path: Path, refresh: bool = False):
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    _throttle()
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(resp.text)
    return resp.json()


def cik10(cik) -> str:
    return str(int(cik)).zfill(10)


def company_tickers(session=None):
    """Mapping of ticker/title/CIK for current SEC registrants."""
    session = session or _session()
    url = "https://www.sec.gov/files/company_tickers.json"
    return _get_json(session, url, config.RAW_DIR / "company_tickers.json")


def submissions(cik, session=None, refresh=False):
    session = session or _session()
    c = cik10(cik)
    url = f"https://data.sec.gov/submissions/CIK{c}.json"
    return _get_json(session, url, config.RAW_DIR / "submissions" / f"{c}.json", refresh)


def company_facts(cik, session=None, refresh=False):
    session = session or _session()
    c = cik10(cik)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c}.json"
    return _get_json(session, url, config.RAW_DIR / "facts" / f"{c}.json", refresh)


def sic_is_excluded(sic) -> bool:
    try:
        sic = int(sic)
    except (TypeError, ValueError):
        return False
    return any(lo <= sic <= hi for lo, hi in config.EXCLUDED_SIC_RANGES)
