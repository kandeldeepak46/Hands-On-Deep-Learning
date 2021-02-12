# -*- coding: utf-8 -*-
"""neural_network.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NvW9CW3XKkC32bKz95VHjzcgZ8wuwhFG
"""

# Commented out IPython magic to ensure Python compatibility.
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import tensorflow as tf

# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style="whitegrid", palette="muted", font_scale=1.5)

rcParams["figure.figsize"] = 14, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(tf.__version__)

(x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()

x_train[0]

x_train.shape

x_val.shape

plt.imshow(x_train[0])
plt.grid(False)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot",
]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)

    return x, y


def create_dataset(xs, ys, n_classes=10):
    ys = tf.one_hot(ys, depth=n_classes)
    return (
        tf.data.Dataset.from_tensor_slices((xs, ys))
        .map(preprocess)
        .shuffle(len(ys))
        .batch(128)
    )


train_dataset = create_dataset(x_train, y_train)
val_dataset = create_dataset(x_val, y_val)

model = keras.Sequential(
    [
        keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        keras.layers.Dense(units=256, activation="relu"),
        keras.layers.Dense(units=192, activation="relu"),
        keras.layers.Dense(units=128, activation="relu"),
        keras.layers.Dense(units=10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
history = model.fit(
    train_dataset.repeat(),
    epochs=10,
    steps_per_epoch=500,
    validation_data=val_dataset.repeat(),
    validation_steps=2,
)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.ylim((0, 1))
plt.legend(["train", "test"], loc="upper left")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.ylim((1.5, 2))
plt.legend(["train", "test"], loc="upper left")
plt.show()

predictions = model.predict(val_dataset)

predictions[0]

np.argmax(predictions[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "green"
    else:
        color = "red"

    plt.xlabel(
        "Predicted: {} {:2.0f}% (True: {})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label],
        ),
        color=color,
    )


i = 3
plot_image(i, predictions, y_val, x_val)
