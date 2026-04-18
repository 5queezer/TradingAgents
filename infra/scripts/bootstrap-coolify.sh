#!/usr/bin/env bash
# Create the external Docker networks and volumes that the gateway expects.
# Run this ONCE on the Coolify host before the first deploy.
#
# Usage: ssh root@coolify-host 'bash -s' < bootstrap-coolify.sh

set -euo pipefail

networks=(public)
volumes=(
  traefik-certificates
  redis-data
  coverage-data
  auth-keys
  mcp-memory-data
)

for net in "${networks[@]}"; do
  if docker network inspect "$net" >/dev/null 2>&1; then
    echo "network $net: already exists"
  else
    docker network create "$net"
    echo "network $net: created"
  fi
done

for vol in "${volumes[@]}"; do
  if docker volume inspect "$vol" >/dev/null 2>&1; then
    echo "volume $vol: already exists"
  else
    docker volume create "$vol"
    echo "volume $vol: created"
  fi
done

echo "Done. You can now deploy the gateway via Coolify."
