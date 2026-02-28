import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage.filters import frangi

# ==========================================
# 1. OPTIC DISC LOCALIZATION (K-MEANS)
# ==========================================
def extract_optic_disc_roi(img, margin_ratio=0.2):
    """
    Localizes the optic disc using K-means clustering on the V channel of HSV.
    Crops the image around the localized center.
    """
    img_array = np.array(img, dtype=np.uint8)
    h, w = img_array.shape[:2]
    
    # Convert to HSV and use Value (Brightness) channel
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    
    # Apply heavy blur to remove vessels and noise
    blurred = cv2.GaussianBlur(v_channel, (51, 51), 0)
    
    # Reshape for K-Means (1D array of pixels)
    Z = blurred.reshape((-1, 1))
    Z = np.float32(Z)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3 # Background, Retina, Optic Disc (Brightest)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Find the brightest cluster index
    brightest_idx = np.argmax(center)
    
    # Create a mask for the brightest cluster
    mask = (label == brightest_idx).reshape((h, w)).astype(np.uint8) * 255
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img_array # Fallback to original if no contour found
        
    # Find the largest contour (assumed to be Optic Disc)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box and calculate center
    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(largest_contour)
    cx = x_rect + w_rect // 2
    cy = y_rect + h_rect // 2
    
    # Determine crop size based on image dimensions and margin
    crop_size = int(min(h, w) * (0.3 + margin_ratio)) 
    half_size = crop_size // 2
    
    # Ensure crop boundaries are within image
    x_start = max(0, cx - half_size)
    y_start = max(0, cy - half_size)
    x_end = min(w, cx + half_size)
    y_end = min(h, cy + half_size)
    
    # Adjust if crop size is smaller than expected due to boundaries
    # (Optional: pad image instead if strict size is needed)
    
    cropped_roi = img_array[y_start:y_end, x_start:x_end]
    
    # Resize to standard size for the rest of pipeline
    cropped_roi = cv2.resize(cropped_roi, (224, 224))
    
    return cropped_roi

# ==========================================
# 2. ADVANCED PIPELINE (DIP)
# ==========================================
def apply_frangi_vessel_filter(img):
    """
    Applies Frangi filter to highlight blood vessels based on eigenvalues of Hessian.
    """
    # Assuming input is grayscale (L channel or Green channel)
    # Frangi works best on normalized float images
    img_float = img.astype(float) / 255.0
    # Invert so vessels are bright structures on dark background (fundus specific)
    img_inv = 1.0 - img_float 
    
    # Apply filter
    vesselness = frangi(img_inv, sigmas=range(1, 4, 1), alpha=0.5, beta=0.5, black_ridges=False)
    
    # Normalize back to 0-255 uint8
    if vesselness.max() > 0:
        vesselness = (vesselness / vesselness.max()) * 255.0
    
    return vesselness.astype(np.uint8)

def apply_grahams_contrast(l_channel):
    """
    Applies a localized contrast enhancement (approximation of Graham's).
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(l_channel)

def advanced_preprocessing_pipeline(img):
    """
    Denoise -> Green channel extract -> CLAHE -> Frangi -> Graham -> Unsharp -> Z-score
    """
    img = np.array(img, dtype=np.uint8)
    
    # 0. K-Means Optic Disc ROI Extraction (Cropping)
    # Note: Applying this directly to all training images might remove context if Papilledema affects peripheral vessels. 
    # Usually, Papilledema is strictly optic disc swelling, so ROI is highly effective.
    img = extract_optic_disc_roi(img)
    
    # 1. Non-local means denoising (reduces noise before enhancement)
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 2. Extract Green Channel (Provides best contrast for vessels/exudates in fundus)
    b, g, r = cv2.split(denoised)
    
    # 3. CLAHE (Local Contrast) on Green Channel
    clahe_g = apply_grahams_contrast(g)
    
    # 4. Frangi Vessel Filter
    # We apply Frangi to highlight vessels, then overlay it.
    frangi_g = apply_frangi_vessel_filter(clahe_g)
    
    # 5. Merge enhanced green with original R and B, or reconstruct a pseudo-RGB
    # Here, we blend the Frangi enhanced vessels back into the CLAHE enhanced Green channel
    enhanced_g = cv2.addWeighted(clahe_g, 0.7, frangi_g, 0.3, 0)
    
    # 6. Reconstruct LAB to apply further contrast/color balance if needed, or stick to pseudo RGB
    # We will reconstruct pseudo-RGB using original R, B, and Enhanced G
    pseudo_rgb = cv2.merge((b, enhanced_g, r))
    
    # 7. Unsharp Masking (Sharpening edges)
    gaussian = cv2.GaussianBlur(pseudo_rgb, (9, 9), 10.0)
    unsharp = cv2.addWeighted(pseudo_rgb, 1.5, gaussian, -0.5, 0, pseudo_rgb)
    
    # 8. Z-Score normalization (optional per-image, usually handled by model preprocessing)
    # We return uint8 here so it plays nicely with Keras ImageDataGenerator / tf.data
    # Z-score will be handled mapped later during tf.data pipeline
    
    return unsharp

# ==========================================
# 3. VISUALIZATION HELPERS
# ==========================================
def compare_raw_vs_advanced(image_path):
    """
    Visualizes the difference between Original and Advanced Pipeline.
    """
    if not os.path.exists(image_path):
         print(f"File not found: {image_path}")
         return
         
    original_bgr = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Extract ROI to show what the pipeline sees
    roi_rgb = extract_optic_disc_roi(original_rgb)
    
    # Apply full pipeline
    advanced_result = advanced_preprocessing_pipeline(original_rgb)
    
    # Extract Frangi specifically for visualization
    g_channel = cv2.split(roi_rgb)[1]
    frangi_overlay = apply_frangi_vessel_filter(g_channel)
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(original_rgb)
    axs[0].set_title("1. Raw Original Fundus")
    axs[0].axis('off')
    
    axs[1].imshow(roi_rgb)
    axs[1].set_title("2. K-Means ROI (Optic Disc)")
    axs[1].axis('off')
    
    axs[2].imshow(frangi_overlay, cmap='magma')
    axs[2].set_title("3. Frangi Vessel Filter")
    axs[2].axis('off')
    
    axs[3].imshow(advanced_result)
    axs[3].set_title("4. Advanced Pipeline Final")
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()
