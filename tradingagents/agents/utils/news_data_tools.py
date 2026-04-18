from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    query: Annotated[
        str,
        "Free-text topic to search for (e.g. 'Strait of Hormuz fertilizer', "
        "'semiconductor export controls'). Use this to steer search toward "
        "geopolitical events, supply shocks, regulation, or thematic catalysts "
        "relevant to the instrument. Leave empty for generic macro context.",
    ] = "",
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 15,
) -> str:
    """
    Retrieve global/macroeconomic news, optionally steered by a topic query.
    Uses the configured news_data vendor.

    Call this tool multiple times with different `query` values to cover
    distinct themes (e.g. once for the commodity angle, once for the
    geopolitical angle) rather than relying on a single generic call.
    """
    return route_to_vendor(
        "get_global_news", curr_date, query, look_back_days, limit
    )

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)
