#!/usr/bin/env python
"""Background worker that runs queued TradingAgents jobs from Redis.

Analysis is streamed via LangGraph's updates mode so progress (current node,
completed phases, report sections ready) can be written back to Redis for
pollers.
"""
from __future__ import annotations

import json
import os
import signal
import time
import traceback
from typing import Any

from jobs import JOB_PREFIX, get_client, get_job, pop_job, update_job

REDIS_URL = os.environ["REDIS_URL"]

_shutdown = False


def _sigterm(_sig, _frame):
    global _shutdown
    _shutdown = True
    print("[worker] shutdown requested", flush=True)


signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT, _sigterm)


# Node-name → human-friendly phase label.
_PHASE_LABEL = {
    "Market Analyst": "Market Analyst",
    "Social Analyst": "Social Analyst",
    "News Analyst": "News Analyst",
    "Fundamentals Analyst": "Fundamentals Analyst",
    "tools_market": "Market Analyst (tools)",
    "tools_social": "Social Analyst (tools)",
    "tools_news": "News Analyst (tools)",
    "tools_fundamentals": "Fundamentals Analyst (tools)",
    "Msg Clear Market": "Market Analyst (cleanup)",
    "Msg Clear Social": "Social Analyst (cleanup)",
    "Msg Clear News": "News Analyst (cleanup)",
    "Msg Clear Fundamentals": "Fundamentals Analyst (cleanup)",
    "Bull Researcher": "Bull Researcher (debate)",
    "Bear Researcher": "Bear Researcher (debate)",
    "Research Manager": "Research Manager",
    "Trader": "Trader",
    "Aggressive Analyst": "Risk Debate — Aggressive",
    "Conservative Analyst": "Risk Debate — Conservative",
    "Neutral Analyst": "Risk Debate — Neutral",
    "Portfolio Manager": "Portfolio Manager",
}

# Report section → completion label.
_REPORT_LABEL = {
    "market_report": "Market Analysis",
    "sentiment_report": "Social Sentiment",
    "news_report": "News Analysis",
    "fundamentals_report": "Fundamentals Analysis",
    "investment_plan": "Research Team Plan",
    "trader_investment_plan": "Trader Plan",
    "final_trade_decision": "Final Trade Decision",
}


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


def _merge_delta(state: dict[str, Any], delta: dict[str, Any]) -> None:
    """Merge a LangGraph node delta into accumulated state.

    Most TradingAgents fields use the default replace reducer; `messages` uses
    add_messages which appends. We don't rely on messages downstream, but we
    still concat to keep state coherent.
    """
    for key, val in delta.items():
        if key == "messages" and isinstance(val, list):
            state.setdefault("messages", [])
            state["messages"].extend(val)
        else:
            state[key] = val


def _write_progress(
    client,
    job_id: str,
    node_name: str | None,
    state: dict[str, Any],
    history: list[str],
    step: int,
) -> None:
    reports_done = [
        _REPORT_LABEL[section]
        for section in _REPORT_LABEL
        if state.get(section)
    ]
    debate_round = state.get("investment_debate_state", {}).get("count", 0)
    risk_round = state.get("risk_debate_state", {}).get("count", 0)
    progress = {
        "current_node": node_name,
        "phase": _PHASE_LABEL.get(node_name, node_name) if node_name else None,
        "step": step,
        "reports_done": reports_done,
        "investment_debate_count": debate_round,
        "risk_debate_count": risk_round,
        "recent_nodes": history[-8:],
        "updated_at": int(time.time()),
    }
    update_job(client, job_id, progress=json.dumps(progress))


def run_analysis(payload: dict[str, Any], client, job_id: str) -> dict[str, Any]:
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    cfg = _build_config(payload)
    analysts = payload.get("analysts") or ["market", "social", "news", "fundamentals"]
    ta = TradingAgentsGraph(selected_analysts=analysts, debug=False, config=cfg)

    init_state = ta.propagator.create_initial_state(payload["ticker"], payload["date"])
    args = ta.propagator.get_graph_args()
    args["stream_mode"] = "updates"

    state: dict[str, Any] = dict(init_state)
    history: list[str] = []
    step = 0
    _write_progress(client, job_id, "starting", state, history, step)

    for chunk in ta.graph.stream(init_state, **args):
        for node_name, delta in chunk.items():
            step += 1
            history.append(node_name)
            if delta:
                _merge_delta(state, delta)
            _write_progress(client, job_id, node_name, state, history, step)

    decision = ta.process_signal(state.get("final_trade_decision", ""))
    ta.curr_state = state
    try:
        ta._log_state(payload["date"], state)
    except Exception as exc:  # logging is best-effort
        print(f"[worker] _log_state failed: {exc}", flush=True)

    return {
        "decision": decision,
        "final_trade_decision": state.get("final_trade_decision"),
        "ticker": payload["ticker"],
        "date": payload["date"],
        "analysts": analysts,
        "reports": {
            label: state.get(section)
            for section, label in _REPORT_LABEL.items()
            if state.get(section)
        },
        "steps_executed": step,
        "nodes_visited": history,
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
            result = run_analysis(payload, client, job_id)
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


def _housekeeping(client) -> None:
    """Mark jobs still flagged `running` as errored — their previous worker died."""
    now = int(time.time())
    rescued = 0
    for key in client.scan_iter(match=JOB_PREFIX + "*", count=100):
        if client.hget(key, "state") == "running":
            client.hset(key, mapping={
                "state": "error",
                "error": "worker_restart_detected",
                "finished_at": str(now),
            })
            rescued += 1
    if rescued:
        print(
            f"[worker] housekeeping: marked {rescued} stuck job(s) as errored",
            flush=True,
        )


def main() -> None:
    redis_host = REDIS_URL.split("@")[-1]
    print(f"[worker] starting, redis={redis_host}", flush=True)
    client = get_client(REDIS_URL)
    _housekeeping(client)
    while not _shutdown:
        job_id = pop_job(client, timeout=5)
        if job_id:
            process(job_id)
    print("[worker] stopped", flush=True)


if __name__ == "__main__":
    main()
