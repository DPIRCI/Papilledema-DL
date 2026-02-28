import os
import yaml
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import wandb
from wandb.keras import WandbCallback
import tensorflow_addons as tfa

# Local imports
from models import build_compiled_model
from utils import advanced_preprocessing_pipeline

# 1. Load Configurations
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Setup Strategy (TPU / GPU)
def get_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("✅ TPU Strategy Enabled.")
    except Exception as e:
        print("⚠️ No TPU detected. Falling back to default strategy (GPU/CPU).")
        strategy = tf.distribute.get_strategy()
    return strategy

# 3. K-Fold CV Generator Wrapper
def get_kfold_splits(dataset_dir, n_splits=5):
    """
    Returns image paths and labels split by StratifiedKFold
    """
    # Assuming dataset is in structure: class_name / image.jpg
    filepaths = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))
    
    for i, cls_name in enumerate(class_names):
        cls_dir = os.path.join(dataset_dir, cls_name)
        if os.path.isdir(cls_dir):
            for img in os.listdir(cls_dir):
                filepaths.append(os.path.join(cls_dir, img))
                labels.append(i)
                
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['hyperparameters']['seed'])
    return skf.split(filepaths, labels), filepaths, labels, class_names

# 4. tf.data.Dataset Pipeline Build
def parse_image(filepath, label, num_classes):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, config['hyperparameters']['img_size'])
    # Normal tf.data mapping can't easily run advanced OpenCV directly unless tf.numpy_function
    # But for extreme performance on TPU, basic built-ins are better.
    # To keep your DIP pipeline, we wrap it in py_function.
    
    def process_dip(img_tensor):
        img_np = img_tensor.numpy().astype(np.uint8)
        processed = advanced_preprocessing_pipeline(img_np)
        return processed.astype(np.float32)

    image = tf.py_function(process_dip, [image], tf.float32)
    image.set_shape([*config['hyperparameters']['img_size'], 3])
    
    # Normalize
    image = tf.cast(image, tf.float32) / 255.0
    
    # One-hot encode label
    label_oh = tf.one_hot(label, num_classes)
    return image, label_oh

# CutMix / MixUp Augmentation
def mix_up(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    weights = tf.random.gamma([batch_size], alpha, alpha)
    weights = tf.maximum(weights, 1 - weights)
    weights = tf.reshape(weights, [batch_size, 1, 1, 1])
    
    indices = tf.random.shuffle(tf.range(batch_size))
    images_shuffled = tf.gather(images, indices)
    labels_shuffled = tf.gather(labels, indices)
    
    images = images * weights + images_shuffled * (1 - weights)
    
    weights_labels = tf.reshape(weights, [batch_size, 1])
    labels = labels * weights_labels + labels_shuffled * (1 - weights_labels)
    
    return images, labels

def create_dataset(filepaths, labels, num_classes, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(lambda x, y: parse_image(x, y, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(buffer_size=1024)
        
    dataset = dataset.batch(config['hyperparameters']['batch_size'])
    
    if is_training:
        # Apply strict augmentations (MixUp/Cutout)
        dataset = dataset.map(lambda x, y: mix_up(x, y), num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 5. Training Loop
def train_model(model_name):
    # Setup Wandb
    wandb.init(project=config['wandb']['project_name'], 
               entity=config['wandb']['entity'],
               name=f"{model_name}-KFold",
               config=config)
    
    strategy = get_strategy()
    
    # K-Fold Gen
    # Using 'original_data' or 'train_dir' since cross-validation merges train/val
    eval_dir = config['paths']['train_dir'] 
    splits, filepaths, labels, class_names = get_kfold_splits(eval_dir, n_splits=config['hyperparameters']['k_folds'])
    num_classes = len(class_names)
    
    fold_no = 1
    for train_index, val_index in list(splits):
        print(f"\\n--- Fold {fold_no} ---")
        
        train_paths, train_lbls = filepaths[train_index], labels[train_index]
        val_paths, val_lbls = filepaths[val_index], labels[val_index]
        
        train_ds = create_dataset(train_paths, train_lbls, num_classes, is_training=True)
        val_ds = create_dataset(val_paths, val_lbls, num_classes, is_training=False)
        
        # Build Model in Strategy
        model = build_compiled_model(model_name, strategy, lr=config['hyperparameters']['learning_rate_phase1'])
        
        # Callbacks Phase 1
        callbacks_ph1 = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            WandbCallback()
        ]
        
        print("...Starting Phase 1 (Head Training)...")
        history1 = model.fit(train_ds, 
                             validation_data=val_ds, 
                             epochs=config['hyperparameters']['epochs_phase1'],
                             callbacks=callbacks_ph1)
        
        # Callbacks Phase 2 (Unfreeze)
        print("...Starting Phase 2 (Fine Tuning)...")
        # Unfreeze Top 80% Layers
        base_model = model.layers[1] if model_name != "ViT" else model.layers[0] # Very rough depending on arch injection
        base_model.trainable = True
        fine_tune_at = int(len(base_model.layers) * 0.8)
        for layer in base_model.layers[:fine_tune_at]: 
             layer.trainable = False
             
        # Re-compile
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['hyperparameters']['learning_rate_phase2']),
                      loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=config['focal_loss']['alpha'], gamma=config['focal_loss']['gamma']),
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
                      
        callbacks_ph2 = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tfa.callbacks.SWA(start_epoch=3, swa_learning_rate=1e-5), # SWA Callback integration
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(config['paths']['model_save_dir'], f"{model_name}_fold{fold_no}.keras"),
                                               save_best_only=True),
            WandbCallback()
        ]
        
        model.fit(train_ds, validation_data=val_ds,
                  epochs=config['hyperparameters']['epochs_phase2'],
                  callbacks=callbacks_ph2)
                  
        print(f"--- Fold {fold_no} Completed ---")
        fold_no += 1
        
        # If we only want to test the first fold to save time during dev:
        # break
        
    wandb.finish()

if __name__ == "__main__":
    if not os.path.exists(config['paths']['model_save_dir']):
        os.makedirs(config['paths']['model_save_dir'])
    
    # Train enabled models
    for m in config['models']:
        if m['enabled']:
            print(f"🚀 Launching K-Fold Training for {m['name']}...")
            train_model(m['name'])
