# robot-calibration
Attempt at calibrating the ur10e robot + mocap system for exposed demo.


## Usage 

```bash
python -m robot_calibration <ipt-file> --verbose --out <out-file> --format transforms
```

Notes:
- `ipt-file` should be a numpy matrix with 35 columns (16 for Tb, 16 for Tr, 3 for the xyz of the pen)
- If `ipt-file` is omitted, it will try to read from standard in 
- If `out-file` is omitted, it will write to standard out 
- Valid formats are `transforms` (2 x 16 numpy array) or `params` (json dict with x,y,z,a,p)
- Customize initial guesses by e.g. `--x0 0.001` (to guess a 1mm +x error for the robot base transform).
- Optimizer step size and number of iterations can be changed with `--lr` and `--iters`
