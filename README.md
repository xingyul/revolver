### *REvolveR*: Continuous Evolutionary Models for Robot-to-robot Policy Transfer
Created by <a href="http://xingyul.github.io">Xingyu Liu</a>, <a href="https://www.cs.cmu.edu/~dpathak" target="_blank">Deepak Pathak</a> and <a href="http://www.cs.cmu.edu/~kkitani" target="_blank">Kris Kitani</a> from Carnegie Mellon University.

<img src="https://github.com/xingyul/revolver/blob/master/doc/teaser.jpg" width="60%">

### Citation
If you find our work useful in your research, please cite:

        @inproceedings{liu:2022:revolver,
          title={REvolveR: Continuous Evolutionary Models for Robot-to-robot Policy Transfer},
          author={Liu, Xingyu and Pathak, Deepak and Kitani, Kris},
          booktitle={The International Conference on Machine Learning (ICML)},
          year={2022}
        }

### Abstract

A popular paradigm in robotic learning is to train a policy from scratch for every new robot. This is not only inefficient but also often impractical for complex robots. In this work, we consider the problem of transferring a policy across two different robots with significantly different parameters such as kinematics and morphology. Existing approaches that train a new policy by matching the action or state transition distribution, including imitation learning methods, fail due to
optimal action and/or state distribution being mismatched in different robots. In this paper, we propose a novel method named  REvolveR  of using continuous evolutionary models for robotic policy transfer implemented in a physics simulator. We interpolate between the source robot and the target robot by finding a continuous evolutionary change of robot parameters. An expert policy on the source robot is transferred through training on a sequence of intermediate robots that gradually
evolve into the target robot. Experiments on a physics simulator show that the proposed continuous evolutionary model can effectively transfer the policy across robots and achieve superior sample efficiency on new robots. The proposed method is especially advantageous in sparse reward settings where exploration can be significantly reduced.

### Code for MuJoCo Gym Experiments

The code and scripts for our MuJoCo Gym experiments are in [gym/](https://github.com/xingyul/revolver/blob/master/gym/). Please refer to [gym/README.md](https://github.com/xingyul/revolver/blob/master/gym/README.md) for more details on how to use our code.

### Code for Hand Manipulation Suite Experiments

Coming soon. Stay tuned.
