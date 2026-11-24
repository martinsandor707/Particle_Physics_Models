#!/bin/bash
# végleges kör: a 3. körös kiegészítésekkel (variants_v3.json) a teljes pipeline, majd alakok, ellenőrzés, táblák, paraméterek
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
PY=/home/user/.venv/bin/python
for p in electron muon proton pion; do
  ( ROUND=3 $PY scripts/particle_pipeline.py $p > logs/pipeline_v3_$p.log 2>&1
    $PY scripts/particle_shapes.py $p > logs/shapes_v3_$p.log 2>&1 ) &
done
wait
$PY scripts/verify_results.py > logs/verify_v3.log 2>&1
$PY scripts/extension_params.py > logs/extension_params.log 2>&1
$PY scripts/make_readme_tables.py > results/particles/readme_tables.md 2> logs/readme_tables.err
echo ROUND3_DONE >> logs/verify_v3.log
