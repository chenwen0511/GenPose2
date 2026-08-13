#!/usr/bin/env bash
# 转发：Blender SOPE v2 全量 batch
set -euo pipefail
GENPOSE2_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FACTORY_ROOT="${FACTORY_ROOT:-$(cd "$GENPOSE2_ROOT/../data_factory_blender" && pwd)}"
export GENPOSE2_ROOT
exec bash "$FACTORY_ROOT/blender_factory/run_full_batch.sh" "$@"
