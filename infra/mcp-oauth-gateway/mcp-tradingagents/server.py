#!/usr/bin/env python
"""MCP server exposing TradingAgents as tools.

- Granular data tools call TradingAgents' dataflow layer synchronously.
- Full-graph analysis is queued as a Redis job and handled by worker.py.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from jobs import (
    create_job,
    get_client,
    get_job,
    list_jobs,
    new_job_id,
    update_job,
)

REDIS_URL = os.environ["REDIS_URL"]


def _configure_tradingagents() -> None:
    from tradingagents.dataflows.config import set_config
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG.copy()
    cfg["llm_provider"] = os.environ.get(
        "TRADINGAGENTS_LLM_PROVIDER", cfg.get("llm_provider", "openai")
    )
    if os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
        cfg["deep_think_llm"] = os.environ["TRADINGAGENTS_DEEP_THINK_LLM"]
    if os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM"):
        cfg["quick_think_llm"] = os.environ["TRADINGAGENTS_QUICK_THINK_LLM"]
    set_config(cfg)


_configure_tradingagents()

from tradingagents.dataflows.interface import route_to_vendor  # noqa: E402

_allowed_hosts = [
    h.strip()
    for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",")
    if h.strip()
]
_cors_origins_raw = os.environ.get("MCP_CORS_ORIGINS", "*")
_allowed_origins = (
    ["*"]
    if _cors_origins_raw.strip() == "*"
    else [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
)

mcp = FastMCP(
    "tradingagents",
    host="0.0.0.0",
    port=3000,
    streamable_http_path="/mcp",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=bool(_allowed_hosts),
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins,
    ),
)


# ------------------------- Data tools (sync) -------------------------

@mcp.tool()
def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Retrieve OHLCV price data for a ticker.

    Args:
        symbol: Ticker, preserving exchange suffix (e.g. AAPL, TSM, RHM.DE).
        start_date: yyyy-mm-dd
        end_date: yyyy-mm-dd
    """
    return route_to_vendor("get_stock_data", symbol, start_date, end_date)


@mcp.tool()
def get_indicators(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Retrieve a technical indicator (e.g. rsi, macd, bbands). One per call."""
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results: list[str] = []
    for ind in indicators:
        try:
            results.append(
                route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days)
            )
        except ValueError as exc:
            results.append(str(exc))
    return "\n\n".join(results)


@mcp.tool()
def get_fundamentals(ticker: str, curr_date: str) -> str:
    """Comprehensive fundamental data for a ticker at a given date."""
    return route_to_vendor("get_fundamentals", ticker, curr_date)


@mcp.tool()
def get_balance_sheet(
    ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None
) -> str:
    """Balance sheet (freq: annual|quarterly)."""
    return route_to_vendor("get_balance_sheet", ticker, freq, curr_date)


@mcp.tool()
def get_cashflow(
    ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None
) -> str:
    """Cash flow statement (freq: annual|quarterly)."""
    return route_to_vendor("get_cashflow", ticker, freq, curr_date)


@mcp.tool()
def get_income_statement(
    ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None
) -> str:
    """Income statement (freq: annual|quarterly)."""
    return route_to_vendor("get_income_statement", ticker, freq, curr_date)


@mcp.tool()
def get_news(ticker: str, start_date: str, end_date: str) -> str:
    """Ticker-specific news between start_date and end_date (yyyy-mm-dd)."""
    return route_to_vendor("get_news", ticker, start_date, end_date)


@mcp.tool()
def get_global_news(curr_date: str, look_back_days: int = 7, limit: int = 5) -> str:
    """Global macro news ending on curr_date."""
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)


@mcp.tool()
def get_insider_transactions(ticker: str) -> str:
    """Insider transactions for a ticker."""
    return route_to_vendor("get_insider_transactions", ticker)


# ----------------------- Analysis (async job) -----------------------

@mcp.tool()
def start_analysis(
    ticker: str,
    date: str,
    analysts: Optional[list[str]] = None,
    max_debate_rounds: Optional[int] = None,
    deep_think_llm: Optional[str] = None,
    quick_think_llm: Optional[str] = None,
) -> dict:
    """Queue a full TradingAgents analysis. Returns a job_id for polling.

    analysts defaults to ["market", "social", "news", "fundamentals"].
    Runs take several minutes; use get_analysis_status / get_analysis_result.
    """
    client = get_client(REDIS_URL)
    job_id = new_job_id()
    payload = {
        "kind": "analysis",
        "ticker": ticker,
        "date": date,
        "analysts": analysts or ["market", "social", "news", "fundamentals"],
        "max_debate_rounds": max_debate_rounds,
        "deep_think_llm": deep_think_llm,
        "quick_think_llm": quick_think_llm,
    }
    create_job(client, job_id, payload)
    return {"job_id": job_id, "state": "queued"}


@mcp.tool()
def get_analysis_status(job_id: str) -> dict:
    """Poll analysis job state."""
    job = get_job(get_client(REDIS_URL), job_id)
    if not job:
        return {"error": "not_found", "job_id": job_id}
    return {
        "job_id": job_id,
        "state": job.get("state"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "progress": job.get("progress"),
        "error": job.get("error"),
    }


@mcp.tool()
def get_analysis_result(job_id: str) -> dict:
    """Fetch the full result of a completed analysis job."""
    job = get_job(get_client(REDIS_URL), job_id)
    if not job:
        return {"error": "not_found", "job_id": job_id}
    if job.get("state") != "done":
        return {"error": "not_ready", "state": job.get("state"), "job_id": job_id}
    return {"job_id": job_id, **json.loads(job.get("result", "{}"))}


@mcp.tool()
def list_analyses(limit: int = 20) -> list[dict]:
    """List recent analysis jobs (most recent first)."""
    jobs = list_jobs(get_client(REDIS_URL), limit=limit)
    return [
        {
            "job_id": j.get("id"),
            "state": j.get("state"),
            "created_at": j.get("created_at"),
            "payload": json.loads(j.get("payload", "{}")),
        }
        for j in jobs
    ]


@mcp.tool()
def cancel_analysis(job_id: str) -> dict:
    """Cancel a queued job. Running jobs continue to completion."""
    client = get_client(REDIS_URL)
    job = get_job(client, job_id)
    if not job:
        return {"error": "not_found"}
    if job.get("state") != "queued":
        return {"error": "cannot_cancel", "state": job.get("state")}
    update_job(client, job_id, state="cancelled")
    return {"job_id": job_id, "state": "cancelled"}


@mcp.tool()
def reflect_and_remember(source_job_id: str, position_return: float) -> dict:
    """Queue a reflection on a completed analysis, given realized return."""
    client = get_client(REDIS_URL)
    source = get_job(client, source_job_id)
    if not source or source.get("state") != "done":
        return {"error": "source_job_not_done"}
    refl_id = new_job_id()
    payload = {
        "kind": "reflect",
        "source_job_id": source_job_id,
        "position_return": float(position_return),
    }
    create_job(client, refl_id, payload)
    return {"job_id": refl_id, "state": "queued"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
