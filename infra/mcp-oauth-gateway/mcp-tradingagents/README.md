# mcp-tradingagents

MCP service that exposes [TradingAgents](https://github.com/TauricResearch/TradingAgents) as tools via StreamableHTTP, behind the `mcp-oauth-gateway` OAuth layer.

## Tools

**Granular data (sync):**

- `get_stock_data`, `get_indicators`, `get_fundamentals`
- `get_balance_sheet`, `get_cashflow`, `get_income_statement`
- `get_news`, `get_global_news`, `get_insider_transactions`

**Full-graph analysis (async job queue, Redis-backed):**

- `start_analysis(ticker, date, analysts?, ...)` → `{job_id}`
- `get_analysis_status(job_id)`
- `get_analysis_result(job_id)`
- `list_analyses(limit)`
- `cancel_analysis(job_id)`
- `reflect_and_remember(source_job_id, position_return)`

## Architecture

```
┌─────────────────┐    stdio      ┌────────────────────────┐
│ streamablehttp  │──────────────▶│ server.py (FastMCP)    │
│ proxy :3000     │               │  - data tools (sync)   │
└─────────────────┘               │  - enqueue analysis    │
        ▲                         └──────────┬─────────────┘
        │ HTTP                               │ Redis DB 1
        │                                    ▼
  Traefik (OAuth)                   ┌────────────────────┐
                                    │ worker.py          │
                                    │  TradingAgentsGraph│
                                    │  .propagate()      │
                                    └────────────────────┘
```

## Required env

See `../.env.example` in the gateway root.

Core:

- `REDIS_PASSWORD`, `BASE_DOMAIN`, `MCP_CORS_ORIGINS`
- `OPENAI_API_KEY` (required for default OpenAI provider)
- Optional per provider: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
- Data vendors: `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY` (only if you switch vendors from yfinance)

TradingAgents-specific overrides:

- `TRADINGAGENTS_LLM_PROVIDER` (default `openai`)
- `TRADINGAGENTS_DEEP_THINK_LLM`, `TRADINGAGENTS_QUICK_THINK_LLM`
- `TRADINGAGENTS_MAX_DEBATE_ROUNDS` (default `1`)

## Host

`tradingagents.${BASE_DOMAIN}` — e.g. `tradingagents.vasudev.xyz`.
