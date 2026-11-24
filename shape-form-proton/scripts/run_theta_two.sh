#!/bin/bash
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
for p in electron muon proton pion; do
  ( [ -f results/particles/$p/theta_entrance_two.json ] || /home/user/.venv/bin/python scripts/theta_entrance.py $p two > logs/theta_two_$p.log 2>&1 ) &
done
wait
echo THETA2_DONE >> logs/theta_two_done.log
