#!/bin/bash
# Build OpenDiS/ExaDiS (LLNL, BSD-3) in WSL Ubuntu with the Kokkos OpenMP backend and the python binding.
# Usage (from Windows): wsl -d Ubuntu -- bash /mnt/d/Work/unified-defect-framework/tools/exadis/build_wsl.sh
set -e
export PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12 -q
[ -f $HOME/dd/.managed ] || { rm -rf $HOME/dd; uv venv --managed-python -p 3.12 $HOME/dd --seed -q; touch $HOME/dd/.managed; }
$HOME/dd/bin/pip install -q numpy scipy matplotlib
PY=$HOME/dd/bin/python
INC=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
echo "python include: $INC"; ls $INC/Python.h
[ -d $HOME/OpenDiS ] || git clone --recursive https://github.com/OpenDiS/OpenDiS.git $HOME/OpenDiS
cd $HOME/OpenDiS/core/exadis
git log -1 --format='exadis commit %H %cd'
rm -rf build && mkdir build && cd build
cmake -S .. -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_SERIAL=On -DKokkos_ENABLE_OPENMP=On \
  -DEXADIS_PYTHON_BINDING=On -DPYTHON_EXECUTABLE=$PY -DPYTHON_DEV_INC_DIR=$INC -DEXADIS_FFT=On > cmake.log 2>&1 || { tail -40 cmake.log; exit 1; }
cmake --build . -j 20 > build.log 2>&1 || { grep -m 20 -i error build.log; tail -20 build.log; exit 1; }
tail -3 build.log
find .. -name "pyexadis*.so" | head
