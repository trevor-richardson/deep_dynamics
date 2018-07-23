# deep_dynamics
General data-driven framework for robotic collision detection. <br/>
<br/>
Train a neural network to regress future states given the current state and action. Stochatsic forward passes are used at inference time in order to produce a belief distribution over the future next state of the robot. An exponentially smoothed norm is used between the set of regressed future states and the ground truth collected in the next time-step. <br/> <br/>

<img src="https://github.com/trevor-richardson/deep_dynamics/blob/master/visualizations/deepdynamics.png" width="950"> <br/>

## Demo

<img src="https://github.com/trevor-richardson/deep_dynamics/blob/master/visualizations/sim1logo-_1_.gif" width="950">
<img src="https://github.com/trevor-richardson/deep_dynamics/blob/master/visualizations/sim2logo-_1_.gif" width="950">
<img src="https://github.com/trevor-richardson/deep_dynamics/blob/master/visualizations/sim3logo-_1_.gif" width="950">

---

### machine_learning
Script trains deep dynamics neural networks from data collected.
```
  python train_dd.py
```
\
Visually evaluate the performance of trained deep dynamics model.
```
  python evaluate_dd_model.py
```
### vrep_scripts
Script generates state actions pairs for training deep dynamics neural network.
```
  python mtr_bab_coll_data.py
```

### vrep_scenes
The following V-REP scene needs to be running for mtr_bab_coll_data.py.
```
  dd_motor_babbling.ttt
```
\
The following V-REP scene needs to be running for evaluate_dd_model.py
```
  dd_current_scene.ttt
```

### Installing
Update config.ini BASE_DIR with the absolute path to current directory. \
Packages needed to run the code
* numpy
* scipy
* python3
* pytorch
* vrep (vrep has instructions for importing API functions)
