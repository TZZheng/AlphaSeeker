"""
Commodity Tool — EIA API: oil and gas inventory and production reports.

Fetches energy market data from the U.S. Energy Information Administration (EIA) API.
Primary data: weekly petroleum status report (crude inventories, production, imports).

Requires: EIA_API_KEY environment variable (free from https://www.eia.gov/opendata/).
"""

import os
from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Common EIA Series IDs
# ---------------------------------------------------------------------------

EIA_SERIES: Dict[str, Dict[str, str]] = {
    "crude_oil": {
        "PET.WCESTUS1.W":   "Weekly U.S. Ending Stocks of Crude Oil (Thousand Barrels)",
        "PET.WCRFPUS2.W":   "Weekly U.S. Field Production of Crude Oil (Thousand Barrels/Day)",
        "PET.WTTIMUS2.W":   "Weekly U.S. Imports of Crude Oil (Thousand Barrels/Day)",
        "PET.RWTC.D":       "Cushing OK WTI Spot Price ($/Barrel)",
    },
    "natural_gas": {
        "NG.NW2_EPG0_SWO_R48_BCF.W":  "Weekly Natural Gas Storage (Bcf)",
        "NG.RNGWHHD.D":               "Henry Hub Natural Gas Spot Price ($/MMBtu)",
    },
    "gasoline": {
        "PET.WGTSTUS1.W":   "Weekly U.S. Ending Stocks of Motor Gasoline (Thousand Barrels)",
    },
}


def get_series_for_commodity(asset: str) -> Dict[str, str]:
    """
    Maps a commodity name to the relevant EIA series IDs.

    Args:
        asset: Commodity name, e.g. "crude oil", "natural gas".

    Returns:
        Dict mapping series_id → description. Empty dict if commodity
        is not energy-related (e.g. gold, copper).
    """
    ...


def fetch_eia_series(
    series_ids: List[str],
    num_observations: int = 52,
) -> Tuple[str, Dict]:
    """
    Fetches one or more EIA time series and saves as a structured Markdown file.

    Each series is formatted as a table with Date | Value columns, plus metadata
    (description, units, frequency).

    Args:
        series_ids: List of EIA API series IDs.
        num_observations: Number of most recent observations to fetch per series.

    Returns:
        Tuple of (file_path, metadata_dict).
        - file_path: Path to the saved Markdown file under data/.
        - metadata_dict: {series_id: {description, units, latest_value, latest_date}}.

    Raises:
        ValueError: If EIA_API_KEY is not set.
        requests.HTTPError: If the EIA API returns an error.
    """
    ...


def fetch_eia_inventory(
    asset: str,
) -> Tuple[Optional[str], Dict]:
    """
    High-level convenience function called by commodity/nodes.py fetch_eia_data node.
    Maps the asset to relevant EIA series, fetches them, and returns results.

    Gracefully returns (None, {}) for non-energy commodities like gold or copper.

    Args:
        asset: Commodity name, e.g. "crude oil".

    Returns:
        Tuple of (file_path_or_None, metadata_dict).
    """
    ...
