#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

code=./scripts/train.py
mode=train
config='./configs/car_tDBN_bv_2.config' # select one config model
output_dir='/home/hk/benz/project/tDBN/results/car_tDBN_bv_2' # output_dir
exe=~/miniconda3/bin/python3 # set your python interpreter

run_script="$exe $code $mode --config_path=$config --model_dir=$output_dir"
$run_script

