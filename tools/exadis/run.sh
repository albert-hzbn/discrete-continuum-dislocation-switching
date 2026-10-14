#!/bin/bash
# Run a python script under the WSL ExaDiS environment: run.sh <dir> <script> [args]
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} OMP_PROC_BIND=spread OMP_PLACES=threads
export PYTHONPATH=$HOME/OpenDiS/core/exadis/python:$HOME/OpenDiS/python:$HOME/OpenDiS/core/pydis/python:$PYTHONPATH
cd "$1"; shift
exec $HOME/dd/bin/python "$@"
