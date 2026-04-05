"""
Commodity Tool — EIA API: oil and gas inventory and production reports.

Fetches energy market data from the U.S. Energy Information Administration (EIA) API.
Primary data: weekly petroleum status report (crude inventories, production, imports).

Requires: EIA_API_KEY environment variable (free from https://www.eia.gov/opendata/).
"""

import os
import re
from typing import List, Tuple, Dict, Optional

from src.shared.reliability import request_json


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


def _sanitize_eia_error_message(message: str, api_key: str) -> str:
    """Redact credentials from exception text before logging or persisting it."""
    sanitized = message.replace(api_key, "[REDACTED]")
    return re.sub(r"(api_key=)[^&\\s]+", r"\1[REDACTED]", sanitized)


def get_series_for_commodity(asset: str) -> Dict[str, str]:
    """
    Maps a commodity name to the relevant EIA series IDs.

    Args:
        asset: Commodity name, e.g. "crude oil", "natural gas".

    Returns:
        Dict mapping series_id → description. Empty dict if commodity
        is not energy-related (e.g. gold, copper).
    """
    asset_normalized = asset.lower().replace(" ", "_")
    return EIA_SERIES.get(asset_normalized, {})


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
    import requests
    import datetime

    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        raise ValueError("EIA_API_KEY environment variable not set.")

    metadata_dict = {}
    markdown_content = "# EIA Energy Market Data\n\n"

    for series_id in series_ids:
        # v2 API endpoint format: https://api.eia.gov/v2/seriesid/series?api_key=your_key
        url = f"https://api.eia.gov/series/?api_key={api_key}&series_id={series_id}"
        
        try:
            data = request_json(
                url,
                timeout=10,
                ttl_seconds=21600,
                attempts=3,
            )

            if "series" not in data or not data["series"]:
                markdown_content += f"## Series {series_id}\nData unavailable.\n\n"
                continue

            series_data = data["series"][0]
            name = series_data.get("name", series_id)
            units = series_data.get("units", "N/A")
            points = series_data.get("data", [])[:num_observations]

            if not points:
                continue

            latest_date_str = str(points[0][0])
            latest_value = points[0][1]

            metadata_dict[series_id] = {
                "description": name,
                "units": units,
                "latest_value": latest_value,
                "latest_date": latest_date_str,
            }

            markdown_content += f"## {name} ({series_id})\n"
            markdown_content += f"- **Units:** {units}\n"
            markdown_content += f"- **Frequency:** {series_data.get('f', 'N/A')}\n\n"
            markdown_content += "| Date | Value |\n|---|---|\n"

            for pt in points:
                date_str = str(pt[0])
                value = pt[1]
                # Format date YYYYMMDD to YYYY-MM-DD
                if len(date_str) == 8:
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                else:
                    formatted_date = date_str
                markdown_content += f"| {formatted_date} | {value} |\n"

            markdown_content += "\n"

        except Exception as e:
            sanitized_error = _sanitize_eia_error_message(str(e), api_key)
            print(f"Error fetching EIA series {series_id}: {sanitized_error}")
            markdown_content += f"## Series {series_id}\nError fetching data: {sanitized_error}\n\n"
            
    save_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"eia_data_{timestamp}.md")

    with open(file_path, "w") as f:
        f.write(markdown_content)

    return file_path, metadata_dict


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
    series_map = get_series_for_commodity(asset)
    if not series_map:
        return None, {}

    try:
        series_ids = list(series_map.keys())
        return fetch_eia_series(series_ids=series_ids)
    except Exception as e:
        print(f"EIA fetch failed for {asset}: {e}")
        return None, {}
