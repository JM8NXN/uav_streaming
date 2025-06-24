#!/bin/sh
env="UAV"
num_uavs=10
num_agents=$((num_uavs))
controller_num_agents=$((num_uavs))
algo="rmappo"
exp="ours_attn2_5uavs"
seed_max=1
scenario="test"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python onpolicy/scripts/train/train_uav.py --env_name ${env} --algorithm_name ${algo} \
    --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} \
    --seed 1 --n_training_threads 28 --n_rollout_threads 128 --num_uavs ${num_uavs} \
    --ctl_num_mini_batch 120 --exe_num_mini_batch 120 --episode_length 120 --num_env_steps 5000000 --hovering_phase_episodes 20000 \
    --ppo_epoch 3 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
    --wandb_name "jm8nxn-university-of-electro-communications" --use_macro --step_difference 5 --controller_num_agents ${controller_num_agents} \
    --use_centralized_V --use_normalized --use_gnn --use_attn --hidden_size 128 --entropy_coef 0.1 --layer_N 3 --share_policy
done
#--layer_N 2
#  --user_name "jm8nxn-university-of-electro-communications"