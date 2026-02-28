import os
import yaml
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_auc_score, f1_score, precision_recall_curve
import shap

from utils import advanced_preprocessing_pipeline

# 1. Configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class_names = ['Normal', 'Papilledema', 'Pseudopapilledema']

# 2. Metrics & Plots
def evaluate_model(y_true, y_pred, y_probs, model_name="Ensemble"):
    # Acc, Precision, Recall, F1
    print(f"\n✅ {model_name} Evaluation")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # ROC-AUC Multi-class (OVR)
    auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, len(class_names)), y_probs, multi_class='ovr')
    print(f"ROC-AUC (OVR): {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f"{model_name}_cm.png")
    plt.close()
    
    # F1-Score Table
    f1s = f1_score(y_true, y_pred, average=None)
    for c, f1 in zip(class_names, f1s):
        print(f"[{c}] F1-Score: {f1:.4f}")

# 3. Grad-CAM++
def make_gradcam_plusplus_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Grad-CAM++ generates heatmaps that capture multiple object instances or tighter bounds
    (e.g., optic disc vs nearby vessels).
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                last_conv_layer_output, preds = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            # 1st order deriv
            conv_first_grad = tape3.gradient(class_channel, last_conv_layer_output)
        # 2nd order deriv
        conv_second_grad = tape2.gradient(conv_first_grad, last_conv_layer_output)
    # 3rd order deriv
    conv_third_grad = tape1.gradient(conv_second_grad, last_conv_layer_output)

    # Calculate Weights (Grad-CAM++ specific formula)
    global_sum = tf.reduce_sum(last_conv_layer_output, axis=(0, 1, 2))
    alpha_num = conv_second_grad
    alpha_denom = conv_second_grad * 2.0 + conv_third_grad * global_sum
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom

    weights = tf.maximum(conv_first_grad, 0.0)
    alpha_normalization_constant = tf.reduce_sum(alphas, axis=(1, 2))
    
    # Final weight vector
    pooled_grads_plusplus = tf.reduce_sum(weights * alphas, axis=(1, 2))
    pooled_grads_plusplus = pooled_grads_plusplus / alpha_normalization_constant

    last_conv_layer_output = last_conv_layer_output[0]
    
    # Squeeze
    heatmap = tf.reduce_sum(last_conv_layer_output * pooled_grads_plusplus[..., tf.newaxis], axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

# 4. Ensemble Prediction
def ensemble_predict(models, img_arrays):
    """
    Averages predictions from multiple models (e.g. from K-Folds or Different Archs)
    """
    all_preds = [m.predict(img_arrays, verbose=0) for m in models]
    avg_preds = np.mean(all_preds, axis=0) # Shape: (batch_size, num_classes)
    final_pred = np.argmax(avg_preds, axis=1)
    return final_pred, avg_preds

# 5. SHAP Feature Importance (KernelExplainer)
def run_shap_analysis(model, X_train_sample, X_test_sample):
    print("⏳ Running SHAP Analysis (This might take a while on images)...")
    # Wrap model predict to only return probabilities
    def f(X):
        return model.predict(X, verbose=0)
    
    # Background dataset for SHAP to compute expected values
    explainer = shap.KernelExplainer(f, X_train_sample)
    
    # Explain predictions on a small test sample
    shap_values = explainer.shap_values(X_test_sample, nsamples=50) # Extremely expensive
    
    # Plot Image SHAP
    shap.image_plot(shap_values, -X_test_sample) # -X since SHAP likes inverse for display
    plt.savefig("shap_feature_importance.png")
    plt.close()
    print("✅ SHAP Analysis saved to shap_feature_importance.png")
