"""
Supervisor shared type definitions.

Kept in a separate module to avoid circular imports between graph.py and router.py.
Both modules import from here; neither imports from the other at module level.
"""


class EntityType:
    """
    Classifies the dominant topic type of a user research request.
    Used by the router to select which sub-agent(s) to invoke.
    """
    EQUITY    = "equity"       # Single public company (e.g., "Analyze AAPL")
    MACRO     = "macro"        # Nation / macro indicators (e.g., "US interest rate outlook")
    COMMODITY = "commodity"    # Physical asset (e.g., "crude oil", "gold")
