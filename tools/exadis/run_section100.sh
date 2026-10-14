#!/bin/bash
# Section 100 declared runs (sources on the wall faces). Stage 1: D, 0 on seeds 1-3 and the H alpha scan on seed 1.
# Stage 2 (after calibration): run_section100.sh stage2 <alpha>
R=/mnt/d/Work/unified-defect-framework
OUT=$R/results/allocator-3d-wall-2026-10-01
RUN="bash $R/tools/exadis/run.sh $R/tools/exadis vein_channel.py"
export OMP_NUM_THREADS=16 SRC_AT_WALL=1 SIGMA=200e6
if [ "$1" = "stage2" ]; then
  for s in 2 3; do $RUN H $s $2 $OUT 2>&1 | grep -v '^step'; done
  exit
fi
for a in 0.1 0.2 0.3 0.5 1.0; do $RUN H 1 $a $OUT 2>&1 | grep -v '^step'; done
for s in 1 2 3; do $RUN 0 $s 0 $OUT 2>&1 | grep -v '^step'; done
for s in 1 2 3; do $RUN D $s 0 $OUT 2>&1 | grep -v '^step'; done
