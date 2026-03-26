############################
# sdpose (py310)
############################

# https://huggingface.co/teemosliang/SDPose-Wholebody

module load gpumem32gb cuda/13.0.2 cudnn/9.8.0.87-12 miniforge3

echo "location of sdpose: $venvs/sdpose"
conda activate $venvs/sdpose
tools=$base/tools/sdpose
mkdir -p $tools

pip uninstall -y pose-format
git clone -b multiple_support https://github.com/GerrySant/pose.git $tools/pose
cd $tools/pose/src/python
pip install -e .

git clone https://github.com/t-s-liang/SDPose-OOD.git
cd SDPose-OOD
pip install -r requirements.txt

# install multimodalhugs
git clone https://github.com/GerrySant/multimodalhugs.git $tools/multimodalhugs

# pin commit  https://github.com/GerrySant/multimodalhugs/commit/5201c80f27aa70c460e8297a799dc5daccbd1b3b
# to avoid unintentionally breaking the code

(cd $tools/multimodalhugs && git checkout "5201c80f27aa70c460e8297a799dc5daccbd1b3b")

(cd $tools/multimodalhugs && pip install .)