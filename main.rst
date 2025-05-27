CIFAR-10 CNN Comparison
=======================

This script implements and compares four convolutional neural network (CNN) architectures on the CIFAR-10 dataset using TensorFlow and Keras. The models include:

- Simple CNN
- Deep CNN with Batch Normalization
- VGG-like CNN
- MobileNetV2 (pretrained on ImageNet)

The models are trained and evaluated using accuracy and confusion matrices. The results are visualized using matplotlib and seaborn.

Modules
-------

- ``tensorflow``
- ``numpy``
- ``matplotlib``
- ``seaborn``
- ``sklearn.metrics``
- ``gc``

Functions and Models
--------------------

.. function:: preprocess(x, y)

   Preprocesses input data by casting it to float32.

   :param x: Image tensor.
   :param y: Label tensor.
   :return: Preprocessed tensors (x, y).

.. function:: simple_cnn()

   Builds a simple CNN model with two Conv2D layers and two MaxPooling2D layers, followed by a Dense and Dropout layer.

   :returns: Keras ``Sequential`` model.

.. function:: deep_cnn()

   Builds a deeper CNN model with BatchNormalization layers. Includes multiple Conv2D, MaxPooling2D, and Dropout layers.

   :returns: Keras ``Sequential`` model.

.. function:: vgg_like_cnn()

   Builds a CNN model inspired by the VGG architecture. Uses stacked Conv2D layers with dropout and pooling.

   :returns: Keras ``Sequential`` model.

.. function:: mobilenetv2_cifar10()

   Builds a MobileNetV2-based model using pretrained weights on ImageNet. The base model is frozen.

   :returns: Keras ``Model``.

Datasets
--------

- ``X_train``, ``X_test``: Normalized CIFAR-10 image data (32x32).
- ``X_train_resized``, ``X_test_resized``: Resized image data (96x96) for MobileNetV2.
- ``y_train``, ``y_test``: One-hot encoded labels.

Training Pipelines
------------------

- ``train_ds``, ``test_ds``: Datasets for 32x32 models using ``tf.data.Dataset``.
- ``train_ds_r``, ``test_ds_r``: Datasets for MobileNetV2 with 96x96 resized images.

Training Loop
-------------

The following steps are executed for each model:

1. Compile the model using Adam optimizer and categorical crossentropy.
2. Train the model for 40 epochs using training and validation datasets.
3. Evaluate using a confusion matrix.
4. Plot validation accuracy over epochs.

Visualization
-------------

- Confusion matrix for each model using seaborn heatmap.
- Accuracy plots over training epochs.

