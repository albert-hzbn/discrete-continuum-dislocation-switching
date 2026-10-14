#!/bin/bash
# Section 99 declared runs, sequential, 16 OpenMP threads each.
# Stage 1: calibration seed 1 (D, H alpha scan, 0); references for seeds 2, 3 (D, 0).
# Stage 2 (after calibration): H on seeds 2, 3 with the calibrated alpha: run_section99.sh stage2 <alpha>
R=/mnt/d/Work/unified-defect-framework
OUT=$R/results/allocator-3d-2026-09-30
RUN="bash $R/tools/exadis/run.sh $R/tools/exadis vein_channel.py"
export OMP_NUM_THREADS=16
if [ "$1" = "stage2" ]; then
  for s in 2 3; do $RUN H $s $2 $OUT 2>&1 | grep -v '^step'; done
  exit
fi
$RUN H 1 0.3 $OUT 2>&1 | grep -v '^step'
for a in 0.1 0.2 0.4 0.5; do $RUN H 1 $a $OUT 2>&1 | grep -v '^step'; done
for s in 1 2 3; do $RUN 0 $s 0 $OUT 2>&1 | grep -v '^step'; done
for s in 1 2 3; do $RUN D $s 0 $OUT 2>&1 | grep -v '^step'; done
