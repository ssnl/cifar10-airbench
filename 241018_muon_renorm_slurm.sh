#! /bin/bash

set +e
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_1e-05 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 1e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_1e-05 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 1e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_1e-05 -- python 241018_muon_renorm.py muon_sign 1e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_3e-05 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 3e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_3e-05 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 3e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_3e-05 -- python 241018_muon_renorm.py muon_sign 3e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_5e-05 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 5e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_5e-05 -- python 241018_muon_renorm.py muon_post_ns 5e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_5e-05 -- python 241018_muon_renorm.py muon_post_ns_nesterov 5e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_5e-05 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 5e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_5e-05 -- python 241018_muon_renorm.py muon_sign 5e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_8e-05 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 8e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_8e-05 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 8e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_8e-05 -- python 241018_muon_renorm.py muon_sign 8e-05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.0001 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.0001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.0001 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.0001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.0001 -- python 241018_muon_renorm.py muon_sign 0.0001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.0003 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.0003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.0003 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.0003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.0003 -- python 241018_muon_renorm.py muon_sign 0.0003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.0005 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.0005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.0005 -- python 241018_muon_renorm.py muon_post_ns 0.0005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.0005 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.0005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.0005 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.0005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.0005 -- python 241018_muon_renorm.py muon_sign 0.0005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.0008 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.0008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.0008 -- python 241018_muon_renorm.py muon_post_ns 0.0008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.0008 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.0008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.0008 -- python 241018_muon_renorm.py muon_sign 0.0008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.001 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.001 -- python 241018_muon_renorm.py muon_post_ns 0.001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.001 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.001 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.001 -- python 241018_muon_renorm.py muon_sign 0.001 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.003 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.003 -- python 241018_muon_renorm.py muon_post_ns 0.003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.003 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.003 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.003 -- python 241018_muon_renorm.py muon_sign 0.003 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.005 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.005 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.005 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.005 -- python 241018_muon_renorm.py muon_sign 0.005 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.008 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.008 -- python 241018_muon_renorm.py muon_post_ns 0.008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.008 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.008 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.008 -- python 241018_muon_renorm.py muon_sign 0.008 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.01 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.01 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.01 -- python 241018_muon_renorm.py muon_post_ns 0.01 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.01 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.01 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.01 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.01 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.01 -- python 241018_muon_renorm.py muon_sign 0.01 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.03 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.03 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.03 -- python 241018_muon_renorm.py muon_post_ns 0.03 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.03 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.03 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.03 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.03 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.03 -- python 241018_muon_renorm.py muon_sign 0.03 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.05 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.05 -- python 241018_muon_renorm.py muon_post_ns 0.05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.05 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.05 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.05 -- python 241018_muon_renorm.py muon_sign 0.05 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.08 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.08 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.08 -- python 241018_muon_renorm.py muon_post_ns 0.08 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.08 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.08 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.08 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.08 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.08 -- python 241018_muon_renorm.py muon_sign 0.08 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.1 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.1 -- python 241018_muon_renorm.py muon_post_ns 0.1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.1 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.1 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.1 -- python 241018_muon_renorm.py muon_sign 0.1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.3 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.3 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.3 -- python 241018_muon_renorm.py muon_post_ns 0.3 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.3 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.3 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.3 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.3 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.3 -- python 241018_muon_renorm.py muon_sign 0.3 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.5 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.5 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.5 -- python 241018_muon_renorm.py muon_post_ns 0.5 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.5 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.5 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.5 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.5 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.5 -- python 241018_muon_renorm.py muon_sign 0.5 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_0.8 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 0.8 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_0.8 -- python 241018_muon_renorm.py muon_post_ns 0.8 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_0.8 -- python 241018_muon_renorm.py muon_post_ns_nesterov 0.8 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_0.8 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 0.8 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_0.8 -- python 241018_muon_renorm.py muon_sign 0.8 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 adam_b09_1 -- python 241018_muon_renorm.py adam_b09 1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_momentum095_1 -- python 241018_muon_renorm.py muon_momentum095 1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_jb_target_ema_grad_norm2_sqrt_dual_1 -- python 241018_muon_renorm.py muon_norm_jb_target_ema_grad_norm2_sqrt_dual 1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_norm_rms_target_unit_1 -- python 241018_muon_renorm.py muon_norm_rms_target_unit 1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_1 -- python 241018_muon_renorm.py muon_post_ns 1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_post_ns_nesterov_1 -- python 241018_muon_renorm.py muon_post_ns_nesterov 1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_nesterov_1 -- python 241018_muon_renorm.py muon_pre_ns_nesterov 1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual_1 -- python 241018_muon_renorm.py muon_pre_ns_norm_jb_target_ema_grad_norm2_sqrt_dual 1 0
python /data/vision/phillipi/contrastive/tongzhou/qrl2/scripts/sbatch.py -g nvidia_h100_80gb_hbm3 nvidia_h100_nvl nvidia_rtx_6000_ada_generation nvidia_a100-sxm4-80gb tesla_v100-sxm2-32gb -e py312 muon_sign_1 -- python 241018_muon_renorm.py muon_sign 1 0
