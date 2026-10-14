#!/bin/bash
# Wait until the D reference runs (seeds 2, 3) have written their results, then run stage 2 (H on seeds 2, 3).
OUT=/mnt/d/Work/unified-defect-framework/results/allocator-3d-2026-09-30
while [ ! -f $OUT/D_s3.json ]; do sleep 60; done
sleep 30
bash /mnt/d/Work/unified-defect-framework/tools/exadis/run_section99.sh stage2 0.1 > $OUT/stage2.log 2>&1
echo done >> $OUT/stage2.log
