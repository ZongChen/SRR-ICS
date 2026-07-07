#!/usr/bin/env bash
set -e

DATA_DIR=${DATA_DIR:-data}
DATASET=${DATASET:-market1501}

python train_clip_ics.py \
  --dataset "${DATASET}" \
  --distance ICS \
  --K_search 60 \
  --K_intra 25 \
  --K_cross 15 \
  --tau_intra 2.5 \
  --beta 0.76 \
  --data-dir "${DATA_DIR}"
