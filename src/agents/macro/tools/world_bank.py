"""
Macro Tool — World Bank API: cross-country economic indicators.

Fetches comparative economic data across countries from the World Bank Open Data API.
Used when the macro topic involves non-US economies or cross-country comparisons.

API Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
No API key required.
"""

from typing import List, Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# Common World Bank Indicator Codes
# ---------------------------------------------------------------------------

WB_INDICATORS: Dict[str, str] = {
    "NY.GDP.MKTP.CD":     "GDP (current US$)",
    "NY.GDP.MKTP.KD.ZG":  "GDP growth (annual %)",
    "FP.CPI.TOTL.ZG":     "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS":     "Unemployment, total (% of labor force)",
    "BN.CAB.XOKA.GD.ZS":  "Current account balance (% of GDP)",
    "GC.DOD.TOTL.GD.ZS":  "Central government debt (% of GDP)",
    "FR.INR.RINR":         "Real interest rate (%)",
    "NE.EXP.GNFS.ZS":     "Exports of goods and services (% of GDP)",
    "NE.IMP.GNFS.ZS":     "Imports of goods and services (% of GDP)",
}

# ISO 3166-1 alpha-3 codes for common economies
COUNTRY_CODES: Dict[str, str] = {
    "US": "USA",
    "China": "CHN",
    "EU": "EUU",
    "Japan": "JPN",
    "UK": "GBR",
    "Germany": "DEU",
    "India": "IND",
    "Brazil": "BRA",
}


import os
import datetime
import requests

def resolve_country_codes(countries: List[str]) -> List[str]:
    """
    Maps natural language country names to World Bank ISO codes.
    Falls back to the input string if no mapping is found.

    Args:
        countries: List of country names or codes, e.g. ["US", "China", "EU"].

    Returns:
        List of ISO alpha-3 codes, e.g. ["USA", "CHN", "EUU"].
    """
    resolved = []
    for c in countries:
        # Simple lookup, case-insensitive
        mapped = COUNTRY_CODES.get(c)
        if not mapped:
            # Try matching keys case-insensitively
            for key, val in COUNTRY_CODES.items():
                if key.lower() == c.lower():
                    mapped = val
                    break
        if mapped:
            resolved.append(mapped)
        else:
            resolved.append(c)  # fallback to raw string
    return resolved


def fetch_world_bank_indicators(
    countries: List[str],
    indicator_codes: Optional[List[str]] = None,
    date_range: str = "2019:2025",
) -> Tuple[str, Dict]:
    """
    Fetches World Bank indicator data for the given countries and saves as Markdown.

    If indicator_codes is not specified, fetches a default set of key macro indicators
    (GDP growth, inflation, unemployment, current account, government debt).

    The output Markdown contains one section per country, each with a table of
    indicators across years.

    Args:
        countries: List of country names (mapped via resolve_country_codes).
        indicator_codes: Optional list of WB indicator codes. Defaults to core set.
        date_range: Year range string, e.g. "2019:2025".

    Returns:
        Tuple of (file_path, metadata_dict).
        - file_path: Path to the saved Markdown file under data/.
        - metadata_dict: {country: {indicator: latest_value}}.

    Raises:
        requests.HTTPError: If the World Bank API returns an error.
    """
    codes = resolve_country_codes(countries)
    if not codes:
        return "", {}
        
    country_str = ";".join(codes)
    
    if not indicator_codes:
        indicator_codes = [
            "NY.GDP.MKTP.KD.ZG",  # GDP growth
            "FP.CPI.TOTL.ZG",     # Inflation
            "SL.UEM.TOTL.ZS",     # Unemployment
            "BN.CAB.XOKA.GD.ZS",  # Current account
            "GC.DOD.TOTL.GD.ZS"   # Gov debt
        ]
        
    metadata_dict = {c: {} for c in codes}
    
    # We will build a nested dict: data[country][indicator][year] = value
    parsed_data = {c: {ind: {} for ind in indicator_codes} for c in codes}
    
    for indicator in indicator_codes:
        print(f"Fetching World Bank indicator: {indicator} for {country_str}")
        url = f"http://api.worldbank.org/v2/country/{country_str}/indicator/{indicator}"
        params = {
            "format": "json",
            "date": date_range,
            "per_page": 500
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        
        json_data = resp.json()
        if len(json_data) < 2:
            continue
            
        records = json_data[1]
        for rec in records:
            country_id = rec.get("countryiso3code")
            # If for some reason ISO3 is not available, try country ID
            if not country_id:
                country_id = rec.get("country", {}).get("id")
                
            if country_id not in parsed_data:
                parsed_data[country_id] = {ind: {} for ind in indicator_codes}
                metadata_dict[country_id] = {}
                
            year = rec.get("date")
            val = rec.get("value")
            
            if val is not None:
                parsed_data[country_id][indicator][year] = round(val, 2)
                
                # Update metadata with latest value
                if indicator not in metadata_dict[country_id]:
                    # Initialize with first valid value seen (which is the most recent because WB sorts desc by year)
                    metadata_dict[country_id][indicator] = round(val, 2)

    # Format into markdown
    markdown_content = "# World Bank Economic Indicators\n\n"
    
    # Collect all years seen
    all_years = set()
    for c_data in parsed_data.values():
        for i_data in c_data.values():
            all_years.update(i_data.keys())
    
    sorted_years = sorted(list(all_years), reverse=True)
    
    for country, c_data in parsed_data.items():
        markdown_content += f"## Country: {country}\n"
        
        years_header = " | ".join(sorted_years)
        markdown_content += f"| Indicator | {years_header} |\n"
        
        sep = "|---|" + "|".join(["---" for _ in sorted_years]) + "|\n"
        markdown_content += sep
        
        for indicator in indicator_codes:
            ind_name = WB_INDICATORS.get(indicator, indicator)
            row = f"| **{ind_name}** |"
            for y in sorted_years:
                val = c_data[indicator].get(y, "N/A")
                row += f" {val} |"
            markdown_content += row + "\n"
            
        markdown_content += "\n"
        
    # Save to file
    save_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(save_dir, f"wb_data_{timestamp}.md")
    
    with open(file_path, "w") as f:
        f.write(markdown_content)
        
    return file_path, metadata_dict
