#!/bin/bash
# Print progress of a running vein_channel.py job (simulated time, steps) via py-spy locals.
P=$(ps -eo pid,args | grep "vein_channel.py $1" | grep -v grep | awk '{print $1}' | head -1)
echo "pid $P"
$HOME/dd/bin/py-spy dump --pid $P --locals 2>&1 | grep -E "^\s+(state|self|tstep|stop_step|N)\b|time|Ep" | cut -c1-300 | head -30
