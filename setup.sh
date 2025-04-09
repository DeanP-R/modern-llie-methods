#!/bin/bash

echo "Cloning LLIE repositories..."
mkdir -p llie_methods
cd llie_methods || exit

# Clone only if not already cloned
[ ! -d "colie" ] && git clone https://github.com/ctom2/colie.git
[ ! -d "SCI" ] && git clone https://github.com/tengyu1998/SCI.git
[ ! -d "Retinexformer" ] && git clone https://github.com/caiyuanhao1998/Retinexformer.git
[ ! -d "Zero-DCE" ] && git clone https://github.com/Li-Chongyi/Zero-DCE.git
cd ..

# Core dependencies
pip install onnxruntime-gpu || pip install onnxruntime
pip install insightface

echo "Checking if guidedFilter is available..."
python3 -c "from cv2.ximgproc import guidedFilter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "guidedFilter not found — installing OpenCV contrib and dependencies..."
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
    pip install --force-reinstall opencv-contrib-python
    pip install insightface scikit-image matplotlib tqdm
    echo "Installation complete. Please manually restart the runtime now."
else
    echo "guidedFilter is already available — skipping OpenCV reinstall."
fi

# -------------------------------
# Retinexformer Setup
# -------------------------------
echo "[Retinexformer] Installing dependencies..."

cd llie_methods/Retinexformer || exit

# Install core Retinexformer requirements
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard
pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips

echo "[Retinexformer] Installing BasicSR..."
python setup.py develop --no_cuda_ext

echo "[Retinexformer] Preparing folders..."
mkdir -p pretrained_weights
mkdir -p Enhancement/DarkFaceTest/input
mkdir -p Enhancement/DarkFaceTest/output

echo "[Retinexformer] Downloading LOL-v1 pretrained weight..."
gdown https://drive.google.com/uc?id=1oPlAzYayqCJUBimXsraWYhKqMZ5jA0SS -O pretrained_weights/LOL_v1.pth

echo "[Retinexformer] Writing DarkFace config..."
cat <<EOL > Options/RetinexFormer_DarkFace.yml
name: "Retinexformer_DarkFace"
model_type: ImageRestorationModel
scale: 1
num_gpu: 1
manual_seed: 0

datasets:
  test:
    name: "DarkFaceTest"
    type: PairedImageDataset
    dataroot_gt: ~
    dataroot_lq: ./Enhancement/DarkFaceTest/input
    io_backend:
      type: disk

network_g:
  type: Retinexformer
  dim: 48
  stage: 3
  num_blocks: [4, 6, 4]
  heads: [1, 2, 4]
  ffn_expansion_factor: 2.66
  bias: False
  LayerNorm_type: WithBias

val:
  save_img: true
  suffix: ~

path:
  pretrain_network_g: pretrained_weights/LOL_v1.pth
  strict_load_g: true
  resume_state: ~
EOL

cd ../../..

echo "[INFO] setup.sh complete."
echo "To run Retinexformer on DarkFace:"
echo "------------------------------------------------------------"
echo "python Enhancement/test_from_dataset.py \\"
echo "    --opt llie_methods/Retinexformer/Options/RetinexFormer_DarkFace.yml \\"
echo "    --weights llie_methods/Retinexformer/pretrained_weights/LOL_v1.pth \\"
echo "    --dataset DarkFace"
echo "------------------------------------------------------------"
