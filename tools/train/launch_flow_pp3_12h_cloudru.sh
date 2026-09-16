#!/usr/bin/env bash
# Backward-compatible path for historical Flow-PP3 launch commands.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/launch_weatherbridge_12h_cloudru.sh" "$@"
