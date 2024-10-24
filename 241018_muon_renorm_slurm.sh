#! /bin/bash

UUID6=$(echo -n $(date +%s%N) | md5sum | fold -w6 | shuf | head -n1)

set +e
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb nvidia_rtx_6000_ada_generation -uuid $UUID6 -q -e py312 adam_1e-07 -- python -u 241018_muon_renorm.py adam 1e-07 0