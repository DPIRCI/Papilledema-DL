import tensorflow as tf
from tensorflow.keras import layers, models

def create_transfer_model(base_model, num_classes=3, model_name="CustomModel"):
    """
    Builds a Transfer Learning model on top of a specified pre-trained base model.
    """
    # 1. Base Model -> No Training Initial Phase (Phase 1)
    base_model.trainable = False

    # 2. Extract Output
    x = base_model.output

    # 3. Custom Head for Classification Setup
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dense(512, activation='relu', name="dense_512")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dense(256, activation='relu', name="dense_256")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    
    # Final Layer
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs, name=model_name)
    return model

def get_model(model_name, input_shape=(224, 224, 3), num_classes=3):
    """
    Returns a configured model (compiled or uncompiled, ready for Strategy Scope).
    Supports ViT via vit_keras if installed.
    """
    if model_name == "MobileNetV2":
        from tensorflow.keras.applications import MobileNetV2
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    
    elif model_name == "DenseNet121":
        from tensorflow.keras.applications import DenseNet121
        base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    
    elif model_name == "EfficientNetB0":
        from tensorflow.keras.applications import EfficientNetB0
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    
    elif model_name == "EfficientNetB3":
        from tensorflow.keras.applications import EfficientNetB3
        base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    
    elif model_name == "EfficientNetB4":
        from tensorflow.keras.applications import EfficientNetB4
        base_model = EfficientNetB4(weights="imagenet", include_top=False, input_shape=input_shape)
        
    elif model_name == "ViT":
        try:
            import vit_keras.vit as vit
            # Using ViT-B16 pre-trained on ImageNet-21k
            base_model = vit.vit_b16(
                image_size=input_shape[0], # Must be 224 or 384
                activation='softmax',
                pretrained=True,
                include_top=False,
                pretrained_top=False,
                classes=num_classes
            )
        except ImportError:
            raise ImportError("Please install vit-keras library via: `pip install vit-keras tensorflow-addons`")
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    if model_name != "ViT":
        model = create_transfer_model(base_model, num_classes=num_classes, model_name=model_name)
    else:
        # vit-keras implementation differs slightly in architecture layering 
        # so we append typical Head layers.
        base_model.trainable = False
        x = base_model.output
        x = layers.Flatten()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs=base_model.input, outputs=outputs, name="ViT-B16-Papilledema")

    return model

def build_compiled_model(model_name, strategy, learning_rate=0.0001):
    """
    Creates and compiles the model inside a TPU/GPU strategy scope with Focal Loss.
    """
    with strategy.scope():
        model = get_model(model_name)
        
        # We're importing addons locally here to ensure the scope resolves
        try:
            import tensorflow_addons as tfa
            loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
                alpha=0.25, gamma=2.0, from_logits=False
            )
            # If using categorical crossentropy target, focal needs integer arrays or one-hot
            # We assume one-hot inputs based on ImageDataGenerator class_mode='categorical'
        except ImportError:
            print("Warning: tensorflow_addons not found. Installing focal loss is necessary. Falling back to CategoricalCrossentropy.")
            loss_fn = 'categorical_crossentropy'

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model
