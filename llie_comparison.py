# !pip install retina-face
# !pip install opencv-python-headless matplotlib tqdm
# llie_face_benchmark.py
"""
Modular LLIE + Face Recognition Pipeline Scaffold
- Runs RetinaFace face detection on Dark Face images before and after enhancement
- Supports multiple LLIE methods (CoLIE, SCI, Retinexformer, etc.)
- Records detection stats and generates visual summary
"""

import os
import cv2
import json
import argparse
import shutil
from retinaface import RetinaFace
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------
# User Configuration
# ---------------------------
llie_method = "CoLIE"  # Options: "SCI", "Retinexformer", "CoLIE", "ZeroDCE", "none", "matlab_script"
darkface_dir = "./darkface/images_test"
output_dir = "./output/enhanced_images"
results_dir = "./results"

# ---------------------------
# Apply LLIE Enhancement
# ---------------------------
def enhance_images(method, input_dir, output_subdir):
    os.makedirs(output_subdir, exist_ok=True)
    for img_path in tqdm(glob(f"{input_dir}/*.jpg")):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)

        # Dummy placeholders — to be replaced with actual LLIE code
        if method == "none":
            enhanced = img
        elif method == "CoLIE":
            # Placeholder for CoLIE enhancement function
            enhanced = img  # TODO: Replace with real enhancement
        elif method == "SCI":
            enhanced = img  # TODO
        elif method == "Retinexformer":
            enhanced = img  # TODO
        elif method == "matlab_script":
            enhanced = img  # TODO (after conversion)
        else:
            raise ValueError("Unknown LLIE method")

        cv2.imwrite(os.path.join(output_subdir, filename), enhanced)

# ---------------------------
# Run RetinaFace Detection
# ---------------------------
def detect_faces(img_dir, result_path):
    stats = {}
    for img_path in tqdm(glob(f"{img_dir}/*.jpg")):
        filename = os.path.basename(img_path)
        try:
            faces = RetinaFace.detect_faces(img_path)
            face_count = len(faces) if isinstance(faces, dict) else 0
        except Exception:
            face_count = 0
        stats[filename] = face_count

    # Save stats
    with open(result_path, "w") as f:
        json.dump(stats, f, indent=2)

# ---------------------------
# Generate Summary Plot
# ---------------------------
def generate_summary(results_dir, output_path):
    summary = {}
    for result_file in glob(f"{results_dir}/*.json"):
        method = os.path.splitext(os.path.basename(result_file))[0]
        with open(result_file) as f:
            stats = json.load(f)
        detected = sum(1 for count in stats.values() if count > 0)
        total = len(stats)
        summary[method] = 100 * detected / total

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(summary.keys(), summary.values(), color='teal')
    plt.ylabel("% of Images with ≥1 Face Detected")
    plt.title("Face Detection Success Rate per LLIE Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ---------------------------
# Main Pipeline
# ---------------------------
if __name__ == "__main__":
    method_output_dir = os.path.join(output_dir, llie_method)
    os.makedirs(method_output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Enhancing images with method: {llie_method}")
    enhance_images(llie_method, darkface_dir, method_output_dir)

    result_json_path = os.path.join(results_dir, f"{llie_method}.json")
    print("Running RetinaFace detection...")
    detect_faces(method_output_dir, result_json_path)

    print("Generating comparison plot...")
    generate_summary(results_dir, "./results/comparison_plot.png")
    print("Done.")
