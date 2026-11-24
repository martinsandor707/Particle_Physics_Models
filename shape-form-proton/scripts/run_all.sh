#!/bin/bash
# Mind a négy részecske elemzése a háttérben, a modellkvótától függetlenül; minden lépés folytatható (újraindításkor kihagyja a kész lépéseket).
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
PY=/home/user/.venv/bin/python
for p in electron muon proton; do
  nohup $PY scripts/particle_pipeline.py $p > logs/pipeline_$p.log 2>&1 &
done
for p in electron proton; do
  for bb in "0,1,2,3,4,5,6,7,8,9" "10,11,12,13,14,15,16,17,18,19"; do
    nohup $PY scripts/particle_bodies.py $p $bb > logs/bodies_${p}_${bb%%,*}.log 2>&1 &
  done
done
# pion: csak ha mindkét gyorsítótár elkészült
( until grep -q "done in" logs/build_cache_pion_s1.log && grep -q "done in" logs/build_cache_pion_s2.log; do sleep 30; done
  nohup $PY scripts/particle_pipeline.py pion > logs/pipeline_pion.log 2>&1 &
  for bb in "0,1,2,3,4,5,6,7,8,9" "10,11,12,13,14,15,16,17,18,19"; do
    nohup $PY scripts/particle_bodies.py pion $bb > logs/bodies_pion_${bb%%,*}.log 2>&1 &
  done ) > /dev/null 2>&1 &
echo launched
