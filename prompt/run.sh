#!/bin/bash

DATA_ROOTS=(
  #"data/scannet"
    #"data/fusion_pd00005_scale02_opa07_all"
    "data/pd00005_vox002_rotation-norm"
    "data/pd00005_vox004_rotation-norm"
    "data/pd00005_scale02_opac02_vox002_rotation-norm"
    "data/pd00005_scale04_opac02_vox002_rotation-norm"
    #"data/pd00005_vox002_opacity"
    #"data/pd00005_scale02_opac02_vox002_scale"
    # "data/pd00005_scale02_opac02_vox002_rotation"
    #"data/pd00005_scale02_opac02_vox002_opacity"
    # "data/pd00005_scale04_opac02_vox002_scale"
    # "data/pd00005_scale04_opac02_vox002_rotation"
    #"data/pd00005_scale04_opac02_vox002_opacity"
    # "data/pd00005_scale04_opac04_vox002_scale"
    # "data/pd00005_scale04_opac04_vox002_rotation"
    # "data/pd00001_vox004_opacity"
    # "data/pd0001_vox004_opacity"
    # "data/pd00005_scale02_vox004_opacity"
    # "data/pd00005_vox004_scale-opacity"
    # "data/pd00005_vox002_scale-opacity"
    # "data/pd0001_vox004_scale-opacity"
    # "data/pd0001_vox002_scale-opacity"
    # "data/pd00005_vox004_rotation-opacity"
    # "data/pd00005_vox002_rotation-opacity"
    # "data/pd0001_vox004_rotation-opacity"
)

for ROOT in "${DATA_ROOTS[@]}"
do
  EXP_NAME="scannet-samples100-tf_b-$(basename "$ROOT")"
  echo "Launching training for data_root: $ROOT"

  DATA_ROOT="$ROOT" \
  sh scripts/train.sh \
    -g 1 \
    -d scannet \
    -n "$EXP_NAME" \
    -r false \
    -c semseg-pt-v3m1-0-base \

  LOG_PATH="exp/scannet/${EXP_NAME}/train.log"
  python3 ./gspread/gspread_results.py "$LOG_PATH" "$EXP_NAME" sample100_test
done
