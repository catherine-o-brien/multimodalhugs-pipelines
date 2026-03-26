#! /bin/bash

running_scripts=$(dirname "$0")
base=$running_scripts/../..
base=$(realpath $base)
scripts=$base/scripts

# set to "false" or "true":

dry_run="true"

# set to your desired estimator
estimator="mediapipe" # options: mmposewholebody, mediapipe, openpifpaf, sdpose

# set to your desired dataset
dataset="phoenix" # options: phoenix

model_name="${dataset}_${estimator}"

# best hyperparams found so far

learning_rate="1e-5"
warmup_steps=500
label_smoothing_factor="0.1"
gradient_accumulation_steps=3

. $scripts/running/run_generic.sh


