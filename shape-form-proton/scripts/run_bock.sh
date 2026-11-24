#!/bin/bash
# a Bock-féle irodalmi mélységprofil újratesztelése (proton, pion) a névütközés javítása után; az adat-gyorsítótár első futáskor újraépül
cd /home/user/tamas/shape-form-proton
export MPLBACKEND=Agg
for p in proton pion; do
  ( /home/user/.venv/bin/python scripts/test_literature_variants.py $p > logs/bock_$p.log 2>&1 ) &
done
wait
echo BOCK_DONE >> logs/bock_done.log
