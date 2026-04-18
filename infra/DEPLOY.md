# Deploying TradingAgents + MCP Gateway on Coolify

Target: `tradingagents.<your-domain>` (OAuth-gated MCP endpoint) plus the rest of the `mcp-oauth-gateway` services.

This stack uses several `external: true` Docker volumes and an `external: true` network (`public`) — Coolify does not create these automatically, so you create them once on the host before the first deploy.

## Prerequisites

- Coolify instance reachable (e.g. `coolify.vasudev.xyz`)
- Public DNS pointing `*.<your-domain>` to the Coolify host (Cloudflare proxied is fine)
- GitHub OAuth App created: `https://github.com/settings/developers`
- Redis password, JWT secret, RSA keys (generated via the gateway's `justfile`)

## 1. Host bootstrap (one-time)

SSH into the Coolify host and run:

```bash
curl -fsSL https://raw.githubusercontent.com/5queezer/TradingAgents/feat/mcp-integration/infra/scripts/bootstrap-coolify.sh | bash
```

This creates the external networks and volumes the gateway depends on.

## 2. Create the Coolify application

1. **New Resource → Docker Compose**
2. **Source**: `https://github.com/5queezer/TradingAgents`, branch `feat/mcp-integration` (or `main` once merged)
3. **Base directory**: `/infra/mcp-oauth-gateway`
4. **Compose path**: `docker-compose.yml`
5. **Port**: leave default (Traefik handles routing, no exposed port needed on the Coolify side — the gateway has its own Traefik)

## 3. Environment variables (Coolify UI)

Minimum required:

| Key                         | Notes |
|-----------------------------|-------|
| `BASE_DOMAIN`               | e.g. `vasudev.xyz` |
| `ACME_EMAIL`                | for Let's Encrypt |
| `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | from your GitHub OAuth App |
| `GATEWAY_JWT_SECRET`        | `just generate-jwt-secret` (≥32 chars) |
| `JWT_PRIVATE_KEY_B64`       | `just generate-rsa-keys` |
| `REDIS_PASSWORD`            | `just generate-redis-password` |
| `ALLOWED_GITHUB_USERS`      | comma-sep list or `*` |
| `MCP_CORS_ORIGINS`          | `*` for dev, or your Claude origin |
| `OPENAI_API_KEY`            | required for default OpenAI provider |
| `TRADINGAGENTS_LLM_PROVIDER`| `openai` (default), `anthropic`, `google`, ... |

Optional: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `TRADINGAGENTS_DEEP_THINK_LLM`, `TRADINGAGENTS_QUICK_THINK_LLM`, `TRADINGAGENTS_MAX_DEBATE_ROUNDS`.

## 4. Deploy

Hit **Deploy** in Coolify. First build takes several minutes (TradingAgents + LangGraph + pandas + torch-free dependencies).

## 5. Wire up the MCP client

In Claude Desktop (or any MCP client) add:

```json
{
  "mcpServers": {
    "tradingagents": {
      "url": "https://tradingagents.<your-domain>/mcp",
      "transport": "streamable-http"
    }
  }
}
```

OAuth flow runs against `auth.<your-domain>`.

## 6. Smoke test

Once deployed, you can smoke-test the granular data tool without OAuth from the host:

```bash
docker compose -f infra/mcp-oauth-gateway/docker-compose.yml exec mcp-tradingagents \
  python -c "from tradingagents.dataflows.interface import route_to_vendor; print(route_to_vendor('get_stock_data','NVDA','2026-04-01','2026-04-10'))"
```

Async analysis lifecycle (from any authenticated MCP client):

```
start_analysis(ticker="NVDA", date="2026-04-15")
 → {"job_id": "…", "state": "queued"}
get_analysis_status(job_id="…")
 → {"state": "running" | "done" | "error", ...}
get_analysis_result(job_id="…")
 → {"decision": "BUY|HOLD|SELL ...", ...}
```
