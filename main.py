import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import confusion_matrix, classification_report
from models.hybrid_cnn_rnn import HybridCNNRNN
import time
from datetime import datetime, timedelta

# 1. Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. Preprocess - Normalize to range 0-1
print("Preprocessing data...")
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten label arrays and convert to one-hot encoding
y_train = to_categorical(np.squeeze(y_train), num_classes=10)
y_test = to_categorical(np.squeeze(y_test), num_classes=10)

print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

# 3. Build Hybrid CNN-RNN model
print("\nBuilding Hybrid CNN-RNN model...")
model = HybridCNNRNN(input_shape=(32, 32, 3), num_classes=10)

# 4. Train for 20 epochs
print("\nTraining model for 20 epochs...")
start_time = time.time()
training_start = datetime.now()
print(f"Training started at: {training_start.strftime('%Y-%m-%d %H:%M:%S')}")

model.train(
    train_data=(x_train, y_train),  # Tuple format for Keras
    val_data=(x_test, y_test),
    epochs=20,
    batch_size=32
)

training_end = datetime.now()
training_duration = time.time() - start_time
print(f"\nTraining completed at: {training_end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time: {timedelta(seconds=int(training_duration))} ({training_duration:.2f} seconds)")

# 5. Evaluate and generate predictions
print("\nEvaluating model...")
eval_start = time.time()
loss, accuracy = model.evaluate(x_test, y_test)
eval_duration = time.time() - eval_start
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Evaluation time: {eval_duration:.2f} seconds")

# Generate predictions
print("Generating predictions...")
pred_start = time.time()
y_pred_probs = model.model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
pred_duration = time.time() - pred_start
print(f"Prediction generation time: {pred_duration:.2f} seconds")

# Compute and print metrics
print("\n" + "="*60)
print("CONFUSION MATRIX:")
print("="*60)
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\n" + "="*60)
print("CLASSIFICATION REPORT:")
print("="*60)
print(classification_report(y_true, y_pred))

# Total execution time
total_end = datetime.now()
total_duration = time.time() - start_time
print("\n" + "="*60)
print("EXECUTION SUMMARY:")
print("="*60)
print(f"Started at: {training_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Completed at: {total_end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total execution time: {timedelta(seconds=int(total_duration))} ({total_duration:.2f} seconds)")
