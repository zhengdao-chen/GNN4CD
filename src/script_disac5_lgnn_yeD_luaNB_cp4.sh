#!/bin/bash

# all commands that start with SBATCH contain commands that are just used by SLURM forscheduling
#################
# set a job name
#SBATCH --job-name=J2NB2otf_5YlgNf8J3RS_J2_nl30_ntr5000_RS
#################
# a file for job output, you can check job progress
#SBATCH --output=disac5_lgnn_yeD_luaNB2_nl30_nf8_J2_ntr5000_RS42.out
#################
# a file for errors from the job
#SBATCH --error=disac5_lgnn_yeD_luaNB2_nl30_nf8_J2_ntr5000_RS42.err
#################
# time you think you need; default is one hour
# in minutes
# In this case, hh:mm:ss, select whatever time you want, the less you ask for the
# fasteryour job will run.
# Default is one hour, this example will run in less that 5 minutes.
#SBATCH --time=6-23:00:00
#################
# --gres will give you one GPU, you can ask for more, up to 4 (or how ever many are on the node/card)
#SBATCH --gres gpu:1
# We are submitting to the batch partition
#SBATCH --qos=batch
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=100000
#SBATCH --constraint=gpu_12gb
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=zc1216@nyu.edu

source activate py36
python3 main_lg_lua_otf_cp.py \
--path_logger '' \
--path_gnn '' \
--filename_existing_gnn '' \
--num_examples_train 6000 \
--num_examples_test 300 \
--p_SBM 0.0 \
--q_SBM 0.045 \
--generative_model 'SBM_multiclass' \
--batch_size 1 \
--mode 'train' \
--clip_grad_norm 40.0 \
--num_features 8 \
--num_layers 30 \
--J 2 \
--N_train 400 \
--N_test 400 \
--print_freq 1 \
--n_classes 5 \
--lr 0.004
