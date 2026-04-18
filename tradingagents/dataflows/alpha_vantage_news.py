from .alpha_vantage_common import _make_api_request, format_datetime_for_api

def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
    }

    return _make_api_request("NEWS_SENTIMENT", params)

def get_global_news(
    curr_date,
    query: str = "",
    look_back_days: int = 7,
    limit: int = 50,
) -> dict[str, str] | str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        query: Optional free-text hint. Alpha Vantage's NEWS_SENTIMENT uses a
               fixed `topics` vocabulary, so we best-effort map common keywords
               onto AV topics. Unmapped free-text queries fall back to the
               default macro topics.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).
    """
    from datetime import datetime, timedelta

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    # Rough keyword → AV-topic mapping. Unknown queries fall through to
    # the generic macro topics.
    _AV_TOPIC_MAP = {
        "energy": "energy_transportation",
        "oil": "energy_transportation",
        "gas": "energy_transportation",
        "shipping": "energy_transportation",
        "tanker": "energy_transportation",
        "hormuz": "energy_transportation",
        "earnings": "earnings",
        "ipo": "ipo",
        "merger": "mergers_and_acquisitions",
        "acquisition": "mergers_and_acquisitions",
        "tech": "technology",
        "fed": "economy_monetary",
        "inflation": "economy_monetary",
        "macro": "economy_macro",
        "geopolit": "economy_macro",
        "crisis": "economy_macro",
    }
    topics = "financial_markets,economy_macro,economy_monetary"
    q = (query or "").lower()
    if q:
        mapped = {t for kw, t in _AV_TOPIC_MAP.items() if kw in q}
        if mapped:
            topics = ",".join(sorted(mapped))

    params = {
        "topics": topics,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(curr_date),
        "limit": str(limit),
    }

    return _make_api_request("NEWS_SENTIMENT", params)


def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)