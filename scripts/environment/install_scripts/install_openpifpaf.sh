#! /bin/bash

############################
# openpifpaf (py310)
############################
install_scripts=$(dirname "$0")
environment=$install_scripts/..
scripts=$environment/..
base=$scripts/..
venvs=$base/venvs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $venvs/openpifpaf
tools=$base/tools/openpifpaf
mkdir -p $tools

echo "Tools dir: $tools"

module load gpumem32gb cuda/11.8.0 cudnn/8.7.0.84-11.8 miniforge3

# install fork of pose-format that extends to openpifpaf
pip uninstall -y pose-format
rm -rf $tools/pose
git clone -b new_estimators https://github.com/catherine-o-brien/pose.git $tools/pose
cd $tools/pose/src/python
python -m pip install -e . --no-cache-dir --no-user
cd ../../..

# install multimodalhugs

git clone https://github.com/GerrySant/multimodalhugs.git $tools/multimodalhugs

# pin commit  https://github.com/GerrySant/multimodalhugs/commit/5201c80f27aa70c460e8297a799dc5daccbd1b3b
# to avoid unintentionally breaking the code

(cd $tools/multimodalhugs && git checkout "5201c80f27aa70c460e8297a799dc5daccbd1b3b")

(cd $tools/multimodalhugs && python -m pip install -e . --no-user)

pip install openpifpaf --no-build-isolation --no-cache-dir --no-user

OPENCV_VERSION=$(python - <<'EOF'
import importlib.metadata as m
try:
    print(m.version("opencv-python"))
except m.PackageNotFoundError:
    print(m.version("opencv-python-headless"))
EOF
)

python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install "opencv-python-headless==${OPENCV_VERSION}"

conda deactivate 