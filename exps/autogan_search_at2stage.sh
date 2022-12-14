#!/usr/bin/env bash

python search_mixed_at2stage.py \
--path ./data/few-shot-images/AnimalFace-dog/img \
--eval_path auto-search-mixed \
--im_size 256 \
--gen_model shared_gan \
--dis_model shared_gan \
--controller controller \
--latent_dim 32 \
--gf_dim 16 \
--df_dim 16 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.5 \
--beta2 0.999 \
--fake_ins 0.05 \
--init_type normal \
--n_sample 5000 \
--get_top_eval_img 5000 \
--ctrl_sample_batch 1 \
--num_candidate 10 \
--topk 5 \
--shared_epoch1 15 \
--shared_epoch2 30 \
--change_epoch 15 \
--grow_step 30 \
--max_search_iter 30 \
--ctrl_step1 15 \
--ctrl_step2 30 \
--exp_name autogan_search_at2stageNoCL