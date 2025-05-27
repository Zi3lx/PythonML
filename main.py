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
from tensorflow.keras.layers import Rescaling, Resizing
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.image import resize
import seaborn as sns
import gc

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train_resized = tf.image.resize(X_train, [96, 96]).numpy()
X_test_resized = tf.image.resize(X_test, [96, 96]).numpy()

# Normalize [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train_resized = X_train_resized / 255.0
X_test_resized = X_test_resized / 255.0

# Z-score normalization
mean = np.mean(X_train, axis=(0,1,2), keepdims=True)
std = np.std(X_train, axis=(0,1,2), keepdims=True)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

mean_r = np.mean(X_train_resized, axis=(0,1,2), keepdims=True)
std_r = np.std(X_train_resized, axis=(0,1,2), keepdims=True)
X_train_resized = (X_train_resized - mean_r) / std_r
X_test_resized = (X_test_resized - mean_r) / std_r

# One-hot
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# tf.data.Dataset for 32x32 models
def preprocess(x, y):
    return tf.cast(x, tf.float32), y

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(10000).map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

# tf.data.Dataset for MobileNetV2
train_ds_r = tf.data.Dataset.from_tensor_slices((X_train_resized, y_train)).shuffle(10000).map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds_r = tf.data.Dataset.from_tensor_slices((X_test_resized, y_test)).map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)


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
    base_model = MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    inputs = Input(shape=(96, 96, 3))

    x = Rescaling(1. / 127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Compile models
#simple_cnn(), deep_cnn(), vgg_like_cnn(),
#'Simple CNN', 'Deep CNN w/ BN', 'VGG-like CNN',
models = [
    (simple_cnn(), train_ds, test_ds, 'Simple CNN'),
    (deep_cnn(), train_ds, test_ds, 'Deep CNN'),
    (vgg_like_cnn(), train_ds, test_ds, 'VGG-like CNN'),
    (mobilenetv2_cifar10(), train_ds_r, test_ds_r, 'MobileNetV2')
]

histories = []

for model, train_data, test_data, name in models:
    print(f"\nTraining {name}...")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, epochs=40, validation_data=test_data, verbose=2)
    histories.append((history, name))

    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(np.vstack([y for _, y in test_data]), axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {name}')
    plt.show()

# Plot val accuracy
plt.figure(figsize=(12, 8))
for history, name in histories:
    plt.plot(history.history['val_accuracy'], label=f'{name} Val Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()