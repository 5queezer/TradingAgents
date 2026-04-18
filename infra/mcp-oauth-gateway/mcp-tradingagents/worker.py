#!/usr/bin/env python
"""Background worker that runs queued TradingAgents jobs from Redis."""
from __future__ import annotations

import json
import os
import signal
import time
import traceback
from typing import Any

from jobs import get_client, get_job, pop_job, update_job

REDIS_URL = os.environ["REDIS_URL"]

_shutdown = False


def _sigterm(_sig, _frame):
    global _shutdown
    _shutdown = True
    print("[worker] shutdown requested", flush=True)


signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT, _sigterm)


def _build_config(payload: dict[str, Any]) -> dict[str, Any]:
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG.copy()
    cfg["llm_provider"] = os.environ.get(
        "TRADINGAGENTS_LLM_PROVIDER", cfg.get("llm_provider", "openai")
    )
    cfg["deep_think_llm"] = (
        payload.get("deep_think_llm")
        or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM")
        or cfg.get("deep_think_llm")
    )
    cfg["quick_think_llm"] = (
        payload.get("quick_think_llm")
        or os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM")
        or cfg.get("quick_think_llm")
    )
    if payload.get("max_debate_rounds") is not None:
        cfg["max_debate_rounds"] = int(payload["max_debate_rounds"])
    elif os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS"):
        cfg["max_debate_rounds"] = int(os.environ["TRADINGAGENTS_MAX_DEBATE_ROUNDS"])
    return cfg


def run_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg = _build_config(payload)
    analysts = payload.get("analysts") or ["market", "social", "news", "fundamentals"]
    ta = TradingAgentsGraph(selected_analysts=analysts, debug=False, config=cfg)
    state, decision = ta.propagate(payload["ticker"], payload["date"])
    return {
        "decision": decision,
        "ticker": payload["ticker"],
        "date": payload["date"],
        "analysts": analysts,
        "state_keys": list(state.keys()) if isinstance(state, dict) else [],
    }


def run_reflection(payload: dict[str, Any]) -> dict[str, Any]:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg = _build_config(payload)
    ta = TradingAgentsGraph(debug=False, config=cfg)
    ta.reflect_and_remember(float(payload["position_return"]))
    return {"reflected": True, "source_job_id": payload.get("source_job_id")}


def process(job_id: str) -> None:
    client = get_client(REDIS_URL)
    job = get_job(client, job_id)
    if not job:
        return
    if job.get("state") == "cancelled":
        print(f"[worker] job {job_id} was cancelled, skipping", flush=True)
        return

    update_job(client, job_id, state="running", started_at=str(int(time.time())))
    try:
        payload = json.loads(job.get("payload", "{}"))
        kind = payload.get("kind", "analysis")
        if kind == "reflect":
            result = run_reflection(payload)
        else:
            result = run_analysis(payload)
        update_job(
            client,
            job_id,
            state="done",
            finished_at=str(int(time.time())),
            result=json.dumps(result, default=str),
        )
        print(f"[worker] job {job_id} done ({kind})", flush=True)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[worker] job {job_id} failed: {exc}\n{tb}", flush=True)
        update_job(
            client,
            job_id,
            state="error",
            finished_at=str(int(time.time())),
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    redis_host = REDIS_URL.split("@")[-1]
    print(f"[worker] starting, redis={redis_host}", flush=True)
    client = get_client(REDIS_URL)
    while not _shutdown:
        job_id = pop_job(client, timeout=5)
        if job_id:
            process(job_id)
    print("[worker] stopped", flush=True)


if __name__ == "__main__":
    main()
