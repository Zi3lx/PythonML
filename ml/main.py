import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Add, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.image import resize
import seaborn as sns
import gc

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Najpierw przeskaluj do [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Oblicz średnią i std po kanałach (dla R, G, B osobno)
mean = np.mean(X_train, axis=(0,1,2), keepdims=True)
std = np.std(X_train, axis=(0,1,2), keepdims=True)

# Z-score normalizacja
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Model 1: Simple CNN
def simple_cnn():
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Model 2: Deeper CNN with BatchNorm
def deep_cnn():
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Model 3: VGG-like CNN
def vgg_like_cnn():
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Dropout(0.3),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(),
        Dropout(0.4),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

def mobilenetv2_cifar10():
    base_model = MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = Input(shape=(32, 32, 3))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Compile models
models = [simple_cnn(), deep_cnn(), vgg_like_cnn(), mobilenetv2_cifar10()]
model_names = ['Simple CNN', 'Deep CNN w/ BN', 'VGG-like CNN', 'MobileNetV2']

for model, name in zip(models, model_names):
    print(f"Compiling {name}...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train models
histories = []
for model, name in zip(models, model_names):
    print(f"\nTraining {name}...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        epochs=80,
        validation_data=(X_test, y_test),
        verbose=2,
    )
    histories.append(history)

    # Predict before clearing model
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {name}')
    plt.show()

    # Clear memory
    del model
    K.clear_session()
    gc.collect()

# Plot learning curves
plt.figure(figsize=(12, 8))
for history, name in zip(histories, model_names):
    plt.plot(history.history['val_accuracy'], label=f'{name} Val Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()