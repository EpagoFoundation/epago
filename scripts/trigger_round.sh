#!/usr/bin/env bash
# Open a competition round on the local validator.
#
# The endpoint is bound to loopback, so this only works from inside the
# validator host. The key is read from the environment, never passed on a
# command line where it would sit in shell history and in `ps`.
#
# Triggering more often than ROUND_MIN_INTERVAL_BLOCKS (~2 days) is harmless:
# the tick loop enforces the interval, so extra requests collapse into one
# round when the interval allows. That makes a simple cron entry safe.
set -uo pipefail

BIND=${EPAGO_ROUND_BIND:-127.0.0.1:8919}
KEY=${EPAGO_ROUND_API_KEY:?set EPAGO_ROUND_API_KEY}

code=$(curl -s -o /tmp/epago-round.out -w '%{http_code}' \
  -X POST "http://${BIND}/round" \
  -H "X-Epago-Round-Key: ${KEY}" \
  --max-time 15)

echo "$(date -u '+%F %T') trigger -> HTTP $code $(cat /tmp/epago-round.out 2>/dev/null)"
[ "$code" = "200" ] || [ "$code" = "202" ]
