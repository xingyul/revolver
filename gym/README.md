## MuJoCo Gym Experiments

This document contains information for running MuJoCo Gym experiments reported in the paper. The current directory contains code and command scripts for our REvolveR and the two baselines: training from scratch and direct policy transfer.

### Definition of MuJoCo Source and Target Models

The definition of the MuJoCo models of the source and target robot are in `make_generalized_envs_*.py`. To visualize the models without any policy control, please use `python make_generalized_envs_ant.py` for *Ant* robots and `python make_generalized_envs_humanoid.py` for *Humanoid* robots.

### Download the Expert Policies on Source Robot

We provide the expert policies trained from on the source robot.
Please download the from the [Google Drive link](https://drive.google.com/drive/folders/1-I17P0cdGSSUmQOYjcpWM8ZaBuL_pCcu?usp=sharing). The RL algorithms and tasks can be inferred from the directory names.

### Launch REvolveR

The command scripts for launching REvolveR are the `command_train_*.sh` files. For example, `command_train_revolver_ant_leg_emerge.sh` is the script for launching REvolveR on *Ant-leg-emerge* task.

After downloading the expert policy models from Google Drive, please change the value of `load_model` in the `command_train_*.sh` files to the directory of the models.

Also, remember to change the value of `policy` to choose to use TD3 or SAC as the RL algorithm.

### Launch Baseline of Training from Scratch

To launch the baseline of training target robot policy from scratch, please use `command_train_from_scratch_*.sh`

### Launch Baseline of Direct Policy Transfer

To launch the baseline of direct policy transfer, please use `command_train_direct_finetune_*.sh`.


### Evaluation Script

The command script for evaluating policy is `command_eval.sh`.
Please change the value of `load_model` in `command_eval.sh` to the prefix of log directories that exclude seed values where the log directory names follow the convention in `command_train_*.sh`.








