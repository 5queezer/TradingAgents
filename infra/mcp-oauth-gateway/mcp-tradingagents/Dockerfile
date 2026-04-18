FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG TRADINGAGENTS_REF=feat/mcp-integration
ARG TRADINGAGENTS_REPO=https://github.com/5queezer/TradingAgents.git

RUN pip install --no-cache-dir \
        "tradingagents @ git+${TRADINGAGENTS_REPO}@${TRADINGAGENTS_REF}" \
        "mcp>=1.2.0" \
        "redis>=5.0.0" \
        "mcp-streamablehttp-proxy @ git+https://github.com/atrawog/mcp-streamablehttp-proxy.git"

WORKDIR /app
COPY server.py worker.py jobs.py /app/

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /home/appuser/.tradingagents \
    && chown -R appuser:appuser /home/appuser /app
USER appuser

EXPOSE 3000

CMD ["mcp-streamablehttp-proxy", "python", "/app/server.py"]
