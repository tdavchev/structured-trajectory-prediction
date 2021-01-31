# Learning Structured Representations of Trajectory Prediction
Author: Todor Davchev, The University of Edinburgh.

This repository relies on python 3.6 and Tensorflow 1.*. It is the associated code to [Learning Structured Representations of Spatial and Interactive Dynamics for Trajectory Prediction in Crowded Scenes](https://arxiv.org/abs/1911.13044).

```bibtex
@article{davchev2020learning,
  title={Learning Structured Representations of Spatial and Interactive Dynamics for Trajectory Prediction in Crowded Scenes},
  author={Davchev, Todor Bozhinov and Burke, Michael and Ramamoorthy, Subramanian},
  journal={IEEE Robotics and Automation Letters},
  year={2020},
  publisher={IEEE}
}
```

### Installation

[Set up and activate an Anaconda environment](https://github.com/CSTR-Edinburgh/mlpractical/blob/mlp2017-8/master/notes/environment-set-up.md), then run the following commands:
```
git clone git@github.com:yadrimz/Trajectory_Prediction_Framework.git
cd Trajectory_Prediction_Framework
conda create -n traj python=3.6
conda activate traj
pip install -r requirements.txt
python setup.py install
python setup.py develop # for developer mode
```

### Training and Inference
Trainign can be done using relatively small CPU power. The example below training is done using MacBook Pro 13 base model. At train time the output is similar to this:
```
0/2700 (epoch 0), train_loss = 0.111, time/batch = 0.001
99/2700 (epoch 3), train_loss = 8.332, time/batch = 0.018
198/2700 (epoch 7), train_loss = 0.538, time/batch = 0.015
```
Inference is then done over all trajectories from the test dataset that fit the chosen criteria.
```
Processed trajectory number :  50 out of  352  trajectories
Processed trajectory number :  100 out of  352  trajectories
Processed trajectory number :  150 out of  352  trajectories
Processed trajectory number :  200 out of  352  trajectories
Processed trajectory number :  250 out of  352  trajectories
Processed trajectory number :  300 out of  352  trajectories
Processed trajectory number :  350 out of  352  trajectories
Total mean error of the model is  0.10521254192652005
```

### Beginner's Guide

New to crowded scene trajectory prediction? [Check this beginner's Guide tutorial](https://github.com/tdavchev/Stochastic-Futures-Prediction).
