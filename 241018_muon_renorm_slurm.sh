#! /bin/bash

set +e
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_5e-05 -- python 241018_muon_renorm.py muon_sign 5e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_8e-05 -- python 241018_muon_renorm.py muon_sign 8e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.0001 -- python 241018_muon_renorm.py muon_sign 0.0001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.0003 -- python 241018_muon_renorm.py muon_sign 0.0003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.0005 -- python 241018_muon_renorm.py muon_sign 0.0005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.0008 -- python 241018_muon_renorm.py muon_sign 0.0008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.001 -- python 241018_muon_renorm.py muon_sign 0.001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.003 -- python 241018_muon_renorm.py muon_sign 0.003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.005 -- python 241018_muon_renorm.py muon_sign 0.005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.008 -- python 241018_muon_renorm.py muon_sign 0.008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.01 -- python 241018_muon_renorm.py muon_sign 0.01 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.03 -- python 241018_muon_renorm.py muon_sign 0.03 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.05 -- python 241018_muon_renorm.py muon_sign 0.05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.08 -- python 241018_muon_renorm.py muon_sign 0.08 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.1 -- python 241018_muon_renorm.py muon_sign 0.1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.3 -- python 241018_muon_renorm.py muon_sign 0.3 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.5 -- python 241018_muon_renorm.py muon_sign 0.5 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_0.8 -- python 241018_muon_renorm.py muon_sign 0.8 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl -e py312 muon_sign_1 -- python 241018_muon_renorm.py muon_sign 1 0
