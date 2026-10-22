"""Configuration: SEC access settings and XBRL tag fallbacks."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "out"

# SEC requires a descriptive User-Agent with contact info. Set EDGAR_USER_AGENT
# in your environment, e.g. "Jane Doe jane@example.com".
USER_AGENT = os.environ.get("EDGAR_USER_AGENT", "")

# SEC fair-access limit is 10 requests per second. Stay well under it.
MAX_REQUESTS_PER_SEC = 5

# Financial and real-estate SIC codes are excluded from the universe.
EXCLUDED_SIC_RANGES = [(6000, 6799)]

# Each concept lists us-gaap tags in priority order. The first tag that has a
# value for a given company-period wins. Extend these as you find gaps.
FLOW_TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "op_income": ["OperatingIncomeLoss"],
    "costs_and_expenses": ["CostsAndExpenses"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}
INSTANT_TAGS = {
    "assets": ["Assets"],
    "equity": ["StockholdersEquity"],
}

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT"}
# Flow facts must cover roughly a year (handles 52/53 week fiscal years).
MIN_DURATION_DAYS = 330
MAX_DURATION_DAYS = 400
