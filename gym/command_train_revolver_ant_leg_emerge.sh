

generalized_env=Ant-v2-leg-emerge

# for training SAC agents
policy=SAC
expl_noise=0.1
load_model=log_SAC_Ant-v2-leg-emerge_interp0_seed123/Jan14_00-09-46/models/iter_3000000_model

# for training TD3 agents
policy=TD3
expl_noise=0.1
load_model=log_TD3_Ant-v2-leg-emerge_interp0_seed0/Jan12_09-22-44/models/iter_3000000_model

command_file=`basename "$0"`
actor_lr=0.0003
critic_lr=0.0003
eval=0
eval_interp_param=-1
render=0
save_freq=100
start_timesteps=10000
interp_start=0.0
interp_end=1.0
r_shaping=1
num_robots=2000
num_interp=500
robot_sample_range=0.1
train_sample_range=0.33
eval_freq=30000
max_timesteps=3000000
which_cuda=0

# change seed accordingly
seed=0
# seed=1
# seed=2
# seed=3
# seed=5

log_dir=logs_${policy}/logs_${generalized_env}/log_${policy}_${generalized_env}_revolver_r_shaping${r_shaping}_num_interp${num_interp}_robot_range${robot_sample_range}_train_range${train_sample_range}_seed${seed}
# log_dir=log_debug

mkdir -p $log_dir


python train_revolver.py \
    --policy $policy \
    --generalized_env $generalized_env \
    --render $render \
    --actor_lr $actor_lr \
    --critic_lr $critic_lr \
    --r_shaping $r_shaping \
    --max_timesteps $max_timesteps \
    --which_cuda $which_cuda \
    --seed $seed \
    --expl_noise $expl_noise \
    --start_timesteps $start_timesteps \
    --load_model $load_model \
    --interp_start $interp_start \
    --interp_end $interp_end \
    --num_robots $num_robots \
    --num_interp $num_interp \
    --robot_sample_range $robot_sample_range \
    --train_sample_range $train_sample_range \
    --eval_freq $eval_freq \
    --save_freq $save_freq \
    --eval $eval \
    --eval_interp_param $eval_interp_param \
    --render $render \
    --log_dir $log_dir \
    --command_file $command_file \
    > $log_dir.txt 2>&1 &
