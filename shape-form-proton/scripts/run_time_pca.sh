#!/bin/bash
# időben szétválasztott illesztés (proton, pion) és záporonkénti ingadozás (mind a négy) – háttérben, folytatható
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
PY=/home/user/.venv/bin/python
for p in proton pion; do
  ( [ -f results/particles/$p/time_split.json ] || $PY scripts/time_split.py $p > logs/timesplit_$p.log 2>&1 ) &
done
for p in electron muon proton pion; do
  ( [ -f results/particles/$p/fluctuations_pca.json ] || $PY scripts/fluctuations_pca.py $p > logs/pca_$p.log 2>&1 ) &
done
wait
echo TP_DONE >> logs/time_pca_done.log
