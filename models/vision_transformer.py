import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize images
x_train, x_test = x_train / 255.0, x_test / 255.0
num_classes = 10

# Define Vision Transformer (ViT) model
def create_vit_model(image_size=32, patch_size=4, num_classes=10, transformer_layers=4, num_heads=8, embed_dim=64):
    inputs = keras.Input(shape=(image_size, image_size, 3))

    # Patch Embedding
    patches = layers.Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    patches = layers.Reshape((-1, embed_dim))(patches)

    # Transformer Encoder
    for _ in range(transformer_layers):
        x = layers.LayerNormalization()(patches)
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
        x = layers.Add()([patches, x])
        x = layers.LayerNormalization()(x)
        x = layers.Dense(units=embed_dim, activation='relu')(x)
        patches = layers.Add()([patches, x])

    # Classification head
    x = layers.GlobalAveragePooling1D()(patches)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model

# Create and compile model
vit_model = create_vit_model()
vit_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
vit_model.fit(x_train, y_train, batch_size=64, epocpyhs=10, validation_data=(x_test, y_test))

# Evaluate model
test_loss, test_acc = vit_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
