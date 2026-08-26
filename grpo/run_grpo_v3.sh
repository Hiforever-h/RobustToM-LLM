#!/usr/bin/env bash

# Compatibility alias. New experiments use natural-CoT + packed LLM Judge.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "run_grpo_v3.sh now forwards to run_grpo_natural.sh." >&2
echo "Use run_grpo_json_v3.sh to reproduce the legacy JSON/few-shot run." >&2
exec bash "${PROJECT_ROOT}/grpo/run_grpo_natural.sh" "$@"
