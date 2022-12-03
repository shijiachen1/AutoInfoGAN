#!/usr/bin/env bash

python eval.py \
--artifacts ./train_results/animalface-dog-gen \
--eval_path animalface-dog-gen \
--im_size 256 \
--iter 10000 \
--n_sample 5000 \
--gen_model shared_gan \
--dis_model shared_gan \
--controller controller \
--latent_dim 256 \
--gf_dim 64 \
--df_dim 64 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.5 \
--beta2 0.999 \
--init_type normal \
--ctrl_sample_batch 1 \
--arch 1 0 0 0 0 0 2 0 2 \
--exp_name derive