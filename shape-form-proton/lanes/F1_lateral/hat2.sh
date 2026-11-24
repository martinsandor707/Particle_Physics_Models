#!/bin/bash
cd /home/user/tamas/shape-form-proton/lanes/F1_lateral
export MPLBACKEND=Agg
/home/user/.venv/bin/python hatprobe.py 2 b_cauchy_expr cauchy,expr 4000 > logs/hat_b_cauchy_expr_n2.log 2>&1
/home/user/.venv/bin/python hatprobe.py 2 f_gauss_cauchy_expr gauss,cauchy,expr 4000 > logs/hat_f_gauss_cauchy_expr_n2.log 2>&1
