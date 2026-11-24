#!/bin/bash
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
for p in electron muon proton pion; do
  ( /home/user/.venv/bin/python scripts/profile_metrics.py $p > logs/profile_$p.log 2>&1 ) &
done
wait
echo PROF_DONE >> logs/profiles_done.log
