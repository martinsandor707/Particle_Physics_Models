#!/bin/bash
# Utómunka részecskénként (alakok), ha a pipeline és a burok-illesztések elkészültek; végül ellenőrzés és README-táblák.
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
PY=/home/user/.venv/bin/python
nbody() { ls results/particles/$1/bodies_E95_EV98/*.json 2>/dev/null | wc -l; }
need() { case $1 in muon) echo 100;; *) echo 60;; esac; }
for p in electron muon proton pion; do
  ( until [ -f results/particles/$p/audit.json ] && [ "$(nbody $p)" -ge "$(need $p)" ]; do sleep 30; done
    $PY scripts/particle_shapes.py $p > logs/shapes_$p.log 2>&1 ) &
done
wait
$PY scripts/verify_results.py > logs/verify.log 2>&1
$PY scripts/make_readme_tables.py > results/particles/readme_tables.md 2> logs/readme_tables.err
echo POST_DONE >> logs/verify.log
