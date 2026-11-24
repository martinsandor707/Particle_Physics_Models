#!/bin/bash
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
for p in electron muon proton pion; do
  ( [ -f results/particles/$p/theta_entrance_beta.json ] || /home/user/.venv/bin/python scripts/theta_entrance.py $p beta > logs/theta_beta_$p.log 2>&1 ) &
done
wait
echo THETAB_DONE >> logs/theta_beta_done.log
