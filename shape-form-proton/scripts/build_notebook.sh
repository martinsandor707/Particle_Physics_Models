#!/bin/bash
# a részletes proton-notebook újraépítése és lefuttatása (nb_src -> 08_shape3d_proton.ipynb)
cd /home/user/tamas/shape-form-proton
/home/user/.venv/bin/python nb_src/build.py 08_shape3d_proton.ipynb cells_1_intro.py cells_2_geometry.py cells_3_likelihood.py \
  cells_4_model.py cells_5_results.py cells_6_components.py cells_7_shape.py cells_8_envelope.py cells_9_s2.py cells_10_end.py > logs/build_nb08.log 2>&1
echo NB_DONE >> logs/build_nb08.log
