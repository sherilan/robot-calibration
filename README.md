# robot-calibration
Attempt at calibrating the ur10e robot + mocap system for exposed demo.

## Model

The system assumes that the xyz world coordinates of the pen tip can be given by

```
[x, y, z, 1]' = Tb * Te * Tr * Tp * [0, 0, 0, 1]'
```

Where:

- `Tb` is the transform from the world frame to the robot base (as given by the mocap system)
- `Te` is the transform that corrects for the error we made when eyeballing the robot base in the mocap system
- `Tr` is the transform from the robot base to the robot end-effector (as given by the robot's forward kinematics)
- `Tp` is the transform from the end-effector to the pen tip (simple translation along z is assumed)

This code estimates `Te` (`a`: rotation around z-axis, `xyz`: xyz translation) and `Tp` (`p`: translation along z) by minimizing the squared euclidean distance between the predicted position of the pen (see equation above) and the observed position (e.g. given by mocap tracking).

## Usage 

```bash
python -m robot_calibration <ipt-file> --verbose --out <out-file> --fmt transforms
```

Notes:
- `ipt-file` should be a numpy matrix with 35 columns (16 for Tb, 16 for Tr, 3 for the xyz of the pen)
- If `ipt-file` is omitted, it will try to read from standard in 
- If `out-file` is omitted, it will write to standard out 
- Valid formats are `transforms` (2 x 16 numpy array) or `params` (json dict with x,y,z,a,p)
- Customize initial guesses by e.g. `--x0 0.001` (to guess a 1mm +x error for the robot base transform).
- Optimizer step size and number of iterations can be changed with `--lr` and `--iters`
