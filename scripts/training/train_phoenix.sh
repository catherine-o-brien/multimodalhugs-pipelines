#! /bin/bash

# calling script needs to set:
# $base
# $dry_run
# $model_name
# $learning_rate
# $gradient_accumulation_steps
# $warmup_steps
# $batch_size
# $label_smoothing_factor

base=$1
dry_run=$2
model_name=$3
estimator_name=$4
learning_rate=$5
gradient_accumulation_steps=$6
warmup_steps=$7
batch_size=$8
label_smoothing_factor=$9

data=$base/data
preprocessed=$data/preprocessed/$estimator_name
scripts=$base/scripts
venvs=$base/venvs
configs=$base/configs
configs_sub=$configs/$model_name

models=$base/models
models_for_estimator=$models/$estimator_name
models_sub=$models_for_estimator/$model_name

mkdir -p $configs
mkdir -p $configs_sub
mkdir -p $models
mkdir -p $models_for_estimator
mkdir -p $models_sub

# skip if checkpoint exists

shopt -s nullglob
checkpoints=("$models_sub"/train/checkpoint*/)

if [ ${#checkpoints[@]} -gt 0 ]; then
    echo "Checkpoint folder exists, skipping"
    exit 0
else
    echo "No checkpoint folder, will start training"
fi

# measure time

SECONDS=0

################################

echo "Python before activating:"
which python

echo "activate path:"
which activate

echo "Executing: source activate $venvs/huggingface"

source activate $venvs/huggingface

echo "Python after activating:"
which python

# necessary?
# export CUDA_VISIBLE_DEVICES=0

# if in doubt, check with:
# echo "CUDA is available:"
# python -c 'import torch; print(torch.cuda.is_available())'

################################

# setup

if [[ $dry_run == "true" ]]; then
    dry_run_arg="--dry-run"
    use_cpu_arg="--use_cpu"
    batch_size="1"
else
    dry_run_arg=""
    use_cpu_arg=""
fi

python $scripts/training/create_config.py \
    --run-name "phoenix" \
    --config-dir $configs_sub \
    --train-metadata-file $preprocessed/rwth_phoenix2014_t.train.tsv \
    --validation-metadata-file $preprocessed/rwth_phoenix2014_t.validation.tsv \
    --test-metadata-file $preprocessed/rwth_phoenix2014_t.test.tsv \
    --new-vocabulary "__dgs__" \
    --learning-rate $learning_rate \
    --gradient-accumulation-steps $gradient_accumulation_steps \
    --warmup-steps $warmup_steps \
    --batch-size $batch_size \
    --label-smoothing-factor $label_smoothing_factor 

# https://github.com/GerrySant/multimodalhugs/issues/50

export HF_HUB_DISABLE_XET=1

# avoid writing to ~/.cache/huggingface

export HF_HOME=$data/huggingface

multimodalhugs-setup \
    --modality "pose2text" \
    --config_path $configs_sub/config_phoenix.yaml \
    --output_dir $models_sub

# training

multimodalhugs-train \
    --task "translation" \
    --config_path $configs_sub/config_phoenix.yaml \
    --setup_path $models_sub/setup \
    --output_dir $models_sub \
    --report_to none $use_cpu_arg

echo "time taken:"
echo "$SECONDS seconds"
