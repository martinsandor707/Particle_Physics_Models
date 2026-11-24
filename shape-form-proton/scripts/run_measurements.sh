#!/bin/bash
# új mérések (független s2 minta): profil-egyezés (profile_metrics.py) és igazság-felbontás (truth_decomposition.py); folytatható
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
PY=/home/user/.venv/bin/python
for p in electron muon proton pion; do
  ( [ -f results/particles/$p/truth_decomposition.json ] || $PY scripts/truth_decomposition.py $p > logs/truth_$p.log 2>&1
    [ -f results/particles/$p/profile_metrics.json ] || $PY scripts/profile_metrics.py $p > logs/profile_$p.log 2>&1 ) &
done
wait
echo MEAS_DONE >> logs/measurements_done.log
