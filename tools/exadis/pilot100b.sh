#!/bin/bash
# Section 100 pilot 2 (reduced models only): sources centred on the wall faces.
R=/mnt/d/Work/unified-defect-framework
OUT=$R/results/allocator-3d-wall-2026-10-01/pilot
export OMP_NUM_THREADS=16 SRC_AT_WALL=1
for S in 200e6 100e6; do
  export SIGMA=$S
  for m in "0 1 0" "H 1 0.1" "H 1 0.3" "H 1 0.5" "H 1 1.0"; do
    bash $R/tools/exadis/run.sh $R/tools/exadis vein_channel.py $m $OUT/wall_S$S 2>&1 | grep -E "wall [0-9]|Error|Traceback"
  done
done
