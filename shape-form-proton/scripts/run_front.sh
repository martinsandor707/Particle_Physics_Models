#!/bin/bash
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
for p in electron muon proton pion; do
  ( /home/user/.venv/bin/python scripts/front_region.py $p > logs/front_$p.log 2>&1 ) &
done
wait
echo FRONT_DONE >> logs/front_done.log
