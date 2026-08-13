#!/usr/bin/env bash
# 转发：v1 多盘点采样合成
set -euo pipefail
GENPOSE2_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FACTORY_ROOT="${FACTORY_ROOT:-$(cd "$GENPOSE2_ROOT/../data_factory_blender" && pwd)}"
export GENPOSE2_ROOT
cd "$FACTORY_ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate genpose2
exec python data_factory/generate_sope_multitray.py "$@"
