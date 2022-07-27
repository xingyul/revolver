## Hand Manipulation Suite Experiments

This document contains information for running Hand Manipulation Suite experiments reported in the paper. The current directory contains code and command scripts for our REvolveR.

### Download the Expert Policies on Source Robot

We provide the expert policies trained from on the source robot.
Please download the from the [Google Drive link](https://drive.google.com/drive/folders/1tuSr8qNA5YtJgTvqlClUHa9YAmINqjFt?usp=sharing). The RL algorithms and tasks can be inferred from the directory names.

### Definition of MuJoCo Source and Target Models

The definition of the MuJoCo models of the source and target robot are in `make_generalized_envs.py`. 
To visualize the models and expert policies, please first download the expert policies files and change the values of `generalized_env` and `policy_file` in the code.
Then use `python make_generalized_envs.py`.

### Launch REvolveR

The command scripts for launching REvolveR are the `command_train_revolver_*.sh` files. For example, `command_train_revolver_hammer.sh` is the script for launching REvolveR on *Hammer* task.

After downloading the expert policy models from Google Drive, please change the value of `policy` and `baseline` in the `command_train_revolver_*.sh` files to the path of the policy and baseline files.

### Visualize Trained Policies

The users can easily develop their own code for visualizing the trained policies based on our implementation in the `make_generalized_envs.py`.







