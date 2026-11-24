#!/bin/bash
# one worker of the fit queue: $1 = comma separated bins
cd /home/user/tamas/shape-form-proton/lanes/E1_bodies
export MPLBACKEND=Agg
PY=/home/user/.venv/bin/python
BINS=$1
# (A) non-parametric bound at the report criterion: surrogate + exact optimisation, started from all fitted bodies
$PY s02_fit.py --bins $BINS --fam knots12 --improve
echo STAGE_A_DONE
# (C) other criteria: surrogate optimisation + exact polish / exact threshold
for crit in "0.90 0.95" "0.90 0.90" "0.80 0.90"; do
  set -- $crit
  $PY s02_fit.py --bins $BINS --fam drop,ellipsoid,gendrop,capsule,knots12 --fe $1 --fev $2 --sur
done
echo STAGE_C_DONE
# (B) further closed-form families at the report criterion
$PY s02_fit.py --bins $BINS --fam capsule_cone,bullet,frustum,superellipsoid_n,superellipsoid,powertip,paraboloid --sur
echo QUEUE_DONE
