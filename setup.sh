# Write the setup script into a shell file

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

echo "Setup complete."
