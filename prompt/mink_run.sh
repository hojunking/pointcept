#!/bin/bash

DATA_ROOTS=(
   #"data/3dgs_pdistance0005_pruned"
   "data/3dgs_attribute_merge_pdis0001_knn5"
   #"data/scale075"
   #"data/3dgs_pdistance00008_pruned"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="scannet-samples-minkunet34c-$(basename "$ROOT")"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-minkunet34c-0-base \
    #-c semseg-minkunet34c-0-base \

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
