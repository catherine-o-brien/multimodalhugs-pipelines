#! /bin/bash

# Default values for parameters if not set

: "${base:="/home/cobrie/scratch/multimodalhugs-examples"}"
: "${dry_run:="false"}"
: "${model_name:="phoenix"}"
: "${estimator_name:="mediapipehands"}"
: "${learning_rate:="5e-05"}"
: "${gradient_accumulation_steps:=1}"
: "${warmup_steps:=0}"
: "${batch_size:=8}"
: "${label_smoothing_factor:="0.0"}"

################################

module load anaconda3

# explicit unloading of GPU modules at this point to use CPU nodes

module unload gpu cuda/12.6.2 cudnn/9.5.1.17-12

scripts=$base/scripts
logs=$base/logs
logs_sub=$logs/$model_name

# logging

mkdir -p $logs $logs_sub

SLURM_DEFAULT_FILE_PATTERN="slurm-%j.out"
SLURM_LOG_ARGS="-o $logs_sub/$SLURM_DEFAULT_FILE_PATTERN -e $logs_sub/$SLURM_DEFAULT_FILE_PATTERN"

echo "##############################################" | tee -a $logs_sub/MAIN
date | tee -a $logs_sub/MAIN
echo "##############################################" | tee -a $logs_sub/MAIN

log_vars() {
  for var in "$@"; do
    # Check if variable is set
    if [[ -v $var ]]; then
      # Use indirect expansion to get value
      echo "${var^^}: ${!var}" | tee -a $logs_sub/MAIN
    fi
  done
}

log_vars base dry_run model_name learning_rate gradient_accumulation_steps warmup_steps batch_size label_smoothing_factor

echo "##############################################" | tee -a $logs_sub/MAIN

# SLURM job args

DRY_RUN_SLURM_ARGS="--cpus-per-task=2 --time=02:00:00 --mem=16G"

TEMP_ARGS_TO_LOAD_DATASET="--cpus-per-task=16 --time=72:00:00 --mem=64G"
SLURM_ARGS_GENERIC="--cpus-per-task=8 --time=24:00:00 --mem=16G"
SLURM_ARGS_TRAIN="--time=36:00:00 --gres=gpu:V100:1 --constraint=GPUMEM32GB --cpus-per-task 8 --mem 16g"
SLURM_ARGS_TRANSLATE="--time=12:00:00 --gres=gpu:V100:1 --constraint=GPUMEM32GB --cpus-per-task 8 --mem 16g"
SLURM_ARGS_EVALUATE="--time=01:00:00 --gres=gpu:V100:1 --constraint=GPUMEM32GB --cpus-per-task 8 --mem 16g"

# if dry run, then all args use generic instances

if [[ $dry_run == "true" ]]; then
  SLURM_ARGS_GENERIC=$DRY_RUN_SLURM_ARGS
  SLURM_ARGS_TRAIN=$DRY_RUN_SLURM_ARGS
  SLURM_ARGS_TRANSLATE=$DRY_RUN_SLURM_ARGS
  SLURM_ARGS_EVALUATE=$DRY_RUN_SLURM_ARGS
fi

# preprocess data with mediapipe hands

id_preprocess=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_GENERIC \
    $SLURM_LOG_ARGS \
    $scripts/preprocessing/$estimator_name/preprocess.sh \
    $base $dry_run
)

echo "  id_preprocess: $id_preprocess | $logs_sub/slurm-$id_preprocess.out" | tee -a $logs_sub/MAIN

# load GPU modules at this point

module load gpu cuda/12.6.2 cudnn/9.5.1.17-12

# HF train (depends on preprocess)

id_train=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_TRAIN \
    --dependency=afterok:$id_preprocess \
    $SLURM_LOG_ARGS \
    $scripts/training/train_phoenix.sh \
    $base $dry_run $model_name $estimator_name \
    $learning_rate $gradient_accumulation_steps $warmup_steps $batch_size $label_smoothing_factor
)

echo "  id_train: $id_train | $logs_sub/slurm-$id_train.out"  | tee -a $logs_sub/MAIN


<<comment
# HF translate + evaluate (depends on train)

id_translate=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_TRANSLATE \
    --dependency=afterok:$id_train \
    $SLURM_LOG_ARGS \
    $scripts/translation/translate_phoenix.sh \
    $base $dry_run $model_name
)

echo "  id_translate: $id_translate | $logs_sub/slurm-$id_translate.out"  | tee -a $logs_sub/MAIN

# evaluate (depends on translate)

id_evaluate=$(
    $scripts/running/sbatch_bare.sh \
    $SLURM_ARGS_EVALUATE \
    --dependency=afterok:$id_translate \
    $SLURM_LOG_ARGS \
    $scripts/evaluation/evaluate.sh \
    $base $dry_run $model_name
)

echo "  id_evaluate: $id_evaluate | $logs_sub/slurm-$id_evaluate.out"  | tee -a $logs_sub/MAIN
comment

