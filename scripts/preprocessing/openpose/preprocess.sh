#! /bin/bash

# calling script needs to set:
# $base
# $dry_run

base=$1
dry_run=$2

scripts=$base/scripts
data=$base/data
venvs=$base/venvs

poses=$data/poses/openpose
preprocessed=$data/preprocess/openpose

mkdir -p $data
mkdir -p $poses $preprocessed

# maybe skip

if [[ -s $preprocessed/rwth_phoenix2014_t.train.tsv ]]; then
    echo "Preprocessed file exists: $preprocessed/openpose/rwth_phoenix2014_t.train.tsv"
    echo "Skipping"
    exit 0
else
    echo "Preprocessed files do not exist yet"
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

################################

if [[ $dry_run == "true" ]]; then
    dry_run_arg="--dry-run"
else
    dry_run_arg=""
fi

python $scripts/preprocessing/openpose/preprocess.py \
    --pose-dir $poses \
    --output-dir $preprocessed \
    --tfds-data-dir $data/tensorflow_datasets $dry_run_arg

# sizes
echo "Sizes of preprocessed TSV files:"

wc -l $preprocessed/*

echo "time taken:"
echo "$SECONDS seconds"
