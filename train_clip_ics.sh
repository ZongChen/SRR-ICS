#!/usr/bin/env bash
set -e

DATA_DIR=${DATA_DIR:-data}

python train_clip_ics.py --dataset market1501 --distance ICS \
  --K_search 60 --K_intra 25 --K_cross 15 --tau_intra 2.5 --beta 0.76 \
  --data-dir "${DATA_DIR}"

# Baseline comparisons used in ablations:
# python train_clip_ics.py --dataset market1501 --distance CAJ --data-dir "${DATA_DIR}"
# python train_clip_ics.py --dataset market1501 --distance UN --data-dir "${DATA_DIR}"
