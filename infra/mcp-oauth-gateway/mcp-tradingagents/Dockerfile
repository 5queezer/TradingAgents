FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Build context is the monorepo root (see docker-compose.yml).
# Install TradingAgents (the outer package) with all deps.
COPY pyproject.toml README.md requirements.txt ./
COPY tradingagents ./tradingagents
COPY cli ./cli
RUN pip install --no-cache-dir .

# MCP server framework + Redis client + streamablehttp proxy
RUN pip install --no-cache-dir \
        "mcp>=1.2.0" \
        "redis>=5.0.0" \
        "mcp-streamablehttp-proxy @ git+https://github.com/atrawog/mcp-streamablehttp-proxy.git"

# Service code
WORKDIR /app
COPY infra/mcp-oauth-gateway/mcp-tradingagents/server.py \
     infra/mcp-oauth-gateway/mcp-tradingagents/worker.py \
     infra/mcp-oauth-gateway/mcp-tradingagents/jobs.py \
     /app/

# Writable cache/results dir for TradingAgents
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.tradingagents \
    && chown -R appuser:appuser /home/appuser /app
USER appuser

EXPOSE 3000

# Default entry: stdio MCP server wrapped by streamablehttp proxy on port 3000.
# Worker container overrides CMD in docker-compose.yml.
CMD ["mcp-streamablehttp-proxy", "python", "/app/server.py"]
