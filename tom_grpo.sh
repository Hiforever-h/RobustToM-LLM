#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${PROJECT_ROOT}/grpo/run_grpo_v3.sh" "$@"
