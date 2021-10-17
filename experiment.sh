#!/usr/bin/env bash

DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
N_RUNS=10

TAG="err=1/5/2 noise=5"

WAVE_DIR=$DIR/data/waves
SIM_DIR=$DIR/data/results/$TAG

export PYTHONPATH="$DIR:$PYTHONPATH"

for wave_file in $(ls $WAVE_DIR); do
  echo "WAVE FILE: $wave_file"
  for i in $(seq $N_RUNS); do
    echo "RUN: $i"
    run_dir="$SIM_DIR/$wave_file/run$i"
    mkdir -p $run_dir
    # 0.01 -> 1cm, 0.05 -> ~3deg, 0.02 -> 2cm
    python $DIR/robot_calibration/sim.py $WAVE_DIR/$wave_file \
      --err_x_scale 0.01 \
      --err_y_scale 0.01 \
      --err_z_scale 0.01 \
      --err_a_scale 0.05 \
      --err_p_scale 0.02 \
      --stationary \
      --speed 0.1 \
      --sim_data $run_dir/sim_data.txt \
      --sim_params $run_dir/sim_params.json

    python $DIR/robot_calibration/est.py $run_dir/sim_data.txt $run_dir/sim_params.json \
      --wandb exposed-robocal \
      --tag $TAG \
      --out $run_dir/estimated_params.json \
      --noise 0.005 \
      --iters 10 \
      --lr 1.0
  done
done

echo "DONE!"
