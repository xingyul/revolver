

policy=SAC
# policy=TD3

command_file=`basename "$0"`
generalized_env=Humanoid-v2-leg-length-mass

load_model=logs_SAC/logs_Humanoid-v2-leg-length-mass/log_SAC_Humanoid-v2-leg-length-mass_revolver_r_shaping1_num_interp500_robot_range0.02_train_range0.33_seed


eval_interp_param=1
max_timesteps=1000000
# max_timesteps=3000000

render=0
which_cuda=0
seed=0


log_dir=log_debug

mkdir -p $log_dir

python eval.py \
    --policy $policy \
    --generalized_env $generalized_env \
    --render $render \
    --eval_interp_param $eval_interp_param \
    --max_timesteps $max_timesteps \
    --which_cuda $which_cuda \
    --seed $seed \
    --load_model $load_model \
    --log_dir $log_dir \
    --command_file $command_file \
    # > $log_dir.txt 2>&1 &
