# %%
"""
# 👁️ Advanced Papilledema Detection Pipeline (Modular Edition)
*   **Author:** [Your Name]
*   **Features:** K-Means Optic Disc Localizer, Frangi Vessel Filters, ViT+EfficientNets, WandB, SWA, TPU Strategy
*   **Explainability:** Grad-CAM++ & SHAP
"""

# %%
# 1. Environment Setup
# Note: %load_ext autoreload and %autoreload 2 are Jupyter/Colab magic commands 
# and cannot run when you use !python main.py. If pasting into a Colab cell, you can uncomment them.

import os
import kagglehub
import shutil

# Install Requirements if needed (Run once)
# !pip install -r requirements.txt
# !pip install timm vit-keras wandb shap tensorflow-addons numpy opencv-python split-folders

# %%
# 2. WandB Authentication 
import wandb
# wandb.login() # Uncomment to login to weights and biases

# %%
# 3. Data Downloader
print("⬇️ Downloading Dataset from Kagglehub...")
path = kagglehub.dataset_download("shashwatwork/identification-of-pseudopapilledema")
print("✅ Complete:", path)

original_dataset_dir = '/content/raw_data'
if os.path.exists(original_dataset_dir):
    shutil.rmtree(original_dataset_dir)
shutil.copytree(path, original_dataset_dir)

# %%
# 4. Data Preparation (Train/Val/Test Split)
# Although we use 5-Fold Stratified CV during training for Train+Val,
# we still separate a pure 15% TEST set for hold-out evaluation.
import splitfolders
output_path = '/content/dataset_split'
if os.path.exists(output_path):
    shutil.rmtree(output_path)

splitfolders.ratio(original_dataset_dir, output=output_path,
                   seed=1337, ratio=(.85, 0.0, .15), group_prefix=None)
                   
print(f"📂 Pure Test Set isolated at: {output_path}/test")
# The 'train' folder here will be used for our 5-Fold CV

# %%
# 5. Pipeline Visualization Test (Raw vs. CLAHE vs. Advanced + Frangi)
from utils import compare_raw_vs_advanced
import random

test_papilledema_dir = os.path.join(output_path, "test", "Papilledema")
if os.path.exists(test_papilledema_dir) and len(os.listdir(test_papilledema_dir)) > 0:
    sample_img = os.path.join(test_papilledema_dir, random.choice(os.listdir(test_papilledema_dir)))
    compare_raw_vs_advanced(sample_img)
else:
    print("Could not find sample image for testing pipeline.")

# %%
# 6. Model Training (TPU Supported, 5-Fold CV, WandB, SWA)
# We can trigger the training directly by importing the train loop
from train import train_model
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Optional: Run only a specific model to save Colab Time
print("\n🔥 STARTING EXPERIMENT RUN (MobileNetV2) 🔥")
train_model("MobileNetV2") # Uncomment to test MobileNet
print("\n🔥 STARTING EXPERIMENT RUN (EfficientNetB4) 🔥")
train_model("EfficientNetB4") # Heavy! Needs GPU/TPU
# train_model("ViT") # vit-keras required

# %%
# 7. Model Evaluation (Grad-CAM++ & SHAP)
# This requires a completed trained model file
'''
from evaluate import evaluate_model, make_gradcam_plusplus_heatmap, run_shap_analysis
import tensorflow as tf
from utils import advanced_preprocessing_pipeline
import cv2
import numpy as np

# A. Load Best Fold Model
# model = tf.keras.models.load_model("/content/models/MobileNetV2_fold1.keras", compile=False)

# B. Get an Image
# img_bgr = cv2.imread(sample_img)
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# processed = advanced_preprocessing_pipeline(img_rgb)
# input_tensor = np.expand_dims(processed / 255.0, axis=0)

# C. Generate Grad-CAM++
# heatmap = make_gradcam_plusplus_heatmap(input_tensor, model, "Conv_1") # Name depends on arch
# final_cam = cv2.addWeighted(processed, 0.5, cv2.resize(np.uint8(255*heatmap), (224,224)), 0.5, 0)
# plt.imshow(final_cam); plt.title("Grad-CAM++"); plt.show()
'''
