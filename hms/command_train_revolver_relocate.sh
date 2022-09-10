


generalized_env=relocate-v0-finger-shrink
policy=log_relocate-v0/iterations/policy_3999.pickle
baseline=log_relocate-v0/iterations/baseline_3999.pickle
command_file=`basename "$0"`

interp_start=0.
interp_end=1.
interp_progression=0.01
num_robots=2000
random_interp_range=0.06
sample_mode=trajectories
num_traj=16
save_freq=1
step_size=0.0001
gamma=0.995
gae=0.97
gpu=7
seed=0
r_shaping=1
log_dir=logs_${generalized_env}/log_${generalized_env}_r_shaping${r_shaping}_random_range${random_interp_range}_seed${seed}
# log_dir=log_debug


mkdir -p $log_dir


python train_revolver.py \
    --generalized_env $generalized_env \
    --command_file $command_file \
    --gpu $gpu \
    --policy $policy \
    --baseline $baseline \
    --r_shaping $r_shaping \
    --interp_start $interp_start \
    --interp_end $interp_end \
    --interp_progression $interp_progression \
    --num_robots $num_robots \
    --random_interp_range $random_interp_range \
    --seed $seed \
    --sample_mode $sample_mode \
    --num_traj $num_traj \
    --save_freq $save_freq \
    --step_size $step_size \
    --gamma $gamma \
    --gae $gae \
    --log_dir $log_dir \
    > $log_dir.txt 2>&1 &
