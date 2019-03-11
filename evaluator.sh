#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

code=./scripts/train.py
mode=evaluate
config=./configs/car_tDBN_vef_1.config # select the config model
output_dir=./results/car_tDBN_vef_1/
result_path=./results/car_tDBN_vef_1/tests  # the directory to save your generated test results.
ckpt_path=./results/car_tDBN_vef_1/voxelnet-191890.tckpt # put the model you want to evluate
exe=~/miniconda3/bin/python3  # set your own python interpreter
test_=Ture  # set True to enerate testing result and set False to evaluate one model
pickle_result=False

run_script="$exe $code $mode --config_path=$config --ckpt_path=$ckpt_path --model_dir=$output_dir --predict_test=$test_ --pickle_result=$pickle_result --result_path=$result_path"
$run_script

