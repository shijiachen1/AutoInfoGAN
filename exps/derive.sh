#!/usr/bin/env bash

python train_search.py \
--path ./data/few-shot-images/AnimalFace-dog/img \
--im_size 256 \
--iter 100000 \
--gen_model shared_gan \
--dis_model shared_gan \
--controller controller \
--latent_dim 256 \
--gf_dim 64 \
--df_dim 64 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.2 \
--beta2 0.9 \
--fake_ins 0.05 \
--init_type normal \
--ctrl_sample_batch 1 \
--arch 0 0 0 0 2 0 1 1 1 1 1 \
--exp_name derive