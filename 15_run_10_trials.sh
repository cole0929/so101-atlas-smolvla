#!/usr/bin/env bash
set -euo pipefail

cd /root/lerobot_project
. /usr/local/Ascend/ascend-toolkit/set_env.sh

exec /opt/smolvla_npu_test/bin/python 15_smolvla_atlas_evaluate.py \
  --motion-key ENABLE_ATLAS_MOTION \
  --trials 10 \
  --duration 60 \
  "$@"
