import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Bidirectional, LSTM,
    BatchNormalization, GlobalAveragePooling1D
)

class HybridCNNRNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        # CNN Layers with BatchNorm and Dropout
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)

        # Reshape for RNN
        x = Reshape((4, 256 * 4))(x)  # Adjusted for 32x32 input
        x = Bidirectional(LSTM(128, return_sequences=False))(x)

        # Dense Layers
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        return model

    def train(self, train_data, val_data, epochs=100, batch_size=32):
        from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

        lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.5, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

        # Unpack tuples if needed
        if isinstance(train_data, tuple) and len(train_data) == 2:
            x_train, y_train = train_data
            x_val, y_val = val_data
            self.model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[lr_scheduler, early_stop]
            )
        else:
            # Assume it's a generator or Dataset
            self.model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[lr_scheduler, early_stop]
            )

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=1)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
