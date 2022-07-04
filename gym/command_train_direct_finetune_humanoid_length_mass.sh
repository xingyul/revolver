

generalized_env=Humanoid-v2-leg-length-mass


# uncomment for training SAC agents
policy=SAC
expl_noise=0.1
load_model=log_SAC_Humanoid-v2_interp0_seed0/Jan18_09-45-46/models/iter_10500000_model

# uncomment for training TD3 agents
# policy=TD3
# expl_noise=0.1
# load_model=log_TD3_Humanoid-v2_interp0/models/iter_11000000_model


command_file=`basename "$0"`
actor_lr=0.0003
critic_lr=0.0003
eval=0
render=0
save_freq=30000
interp=1
start_timesteps=10000
eval_freq=30000
max_timesteps=3000000
which_cuda=0


# change seed accordingly
seed=0
# seed=1
# seed=2
# seed=3
# seed=5


log_dir=logs_${policy}/logs_${generalized_env}/log_${policy}_${generalized_env}_direct_finetune_interp${interp}_seed${seed}
# log_dir=log_debug

mkdir -p $log_dir


python train_original.py \
    --policy $policy \
    --generalized_env $generalized_env \
    --render $render \
    --actor_lr $actor_lr \
    --critic_lr $critic_lr \
    --max_timesteps $max_timesteps \
    --which_cuda $which_cuda \
    --seed $seed \
    --expl_noise $expl_noise \
    --start_timesteps $start_timesteps \
    --load_model $load_model \
    --interp $interp \
    --eval_freq $eval_freq \
    --save_freq $save_freq \
    --eval $eval \
    --render $render \
    --log_dir $log_dir \
    --command_file $command_file \
    > $log_dir.txt 2>&1 &
