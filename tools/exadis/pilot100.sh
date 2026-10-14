#!/bin/bash
# Section 100 pilot (reduced models only; no full-DDD run): does the wall closure change the channel strain?
R=/mnt/d/Work/unified-defect-framework
OUT=$R/results/allocator-3d-wall-2026-10-01/pilot
export OMP_NUM_THREADS=16 L_SRC=3000 Z_SRC=1500
for S in 100e6 70e6; do
  export SIGMA=$S
  for m in "0 1 0" "H 1 0.1" "H 1 0.3" "H 1 0.5"; do
    bash $R/tools/exadis/run.sh $R/tools/exadis vein_channel.py $m $OUT/S$S 2>&1 | grep -E "wall [0-9]|Error|Traceback"
  done
done
