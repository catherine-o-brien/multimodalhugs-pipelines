#! /bin/bash

base="/home/cobrie/scratch/multimodalhugs-examples"
scripts=$base/scripts

# set to "false" or "true":

dry_run="false"

model_name="phoenix"

# best hyperparams found so far

learning_rate="1e-5"
warmup_steps=500
label_smoothing_factor="0.0"
gradient_accumulation_steps=3

. $scripts/running/run_generic.sh
