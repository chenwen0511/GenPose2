#!/usr/bin/env bash
# 转发：Blender SOPE v2 一键渲染（数据 → GenPose2/datasets）
set -euo pipefail
GENPOSE2_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FACTORY_ROOT="${FACTORY_ROOT:-$(cd "$GENPOSE2_ROOT/../data_factory_blender" && pwd)}"
export GENPOSE2_ROOT
exec bash "$FACTORY_ROOT/blender_factory/run.sh" "$@"
