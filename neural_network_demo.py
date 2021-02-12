import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
from pylab import rcParams

try:
    import matplotlib.pyplot as plt
except:
    pass

from matplotlib import rc
from loguru import logger

sns.set(style="whitegrid", palette="muted", font_scale=1.5)

rcParams["figure.figsize"] = 14, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

CLASS_NAMES = [
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


class MyDeepNeuralNetwork(object):
    def __init__(self):
        pass

    def __getitem__(self):
        pass

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

    def train_model(
        self, train_dataset, val_dataset, plot_accuracy=False, plot_loss=False
    ):
        history = self.model.fit(
            train_dataset.repeat(),
            epochs=10,
            steps_per_epoch=500,
            validation_data=val_dataset.repeat(),
            validation_steps=2,
        )

        if plot_accuracy is True:
            plt.plot(history.history["accuracy"])
            plt.plot(history.history["val_accuracy"])
            plt.title("model accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.ylim((0, 1))
            plt.legend(["train", "test"], loc="upper left")
            plt.show()

        if plot_loss is True:
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.ylim((1.5, 2))
            plt.legend(["train", "test"], loc="upper left")
            plt.show()


class MyFashionMNISTData:
    def __init__(self):
        pass

    def preprocess(self, x, y):
        x = tf.cast(x, tf.float32) / 255.0
        y = tf.cast(y, tf.int32)
        return x, y

    def create_dataset(self, xs, ys, n_classes=10):
        self.n_classes = n_classes
        ys = tf.one_hot(ys, depth=self.n_classes)
        return (
            tf.data.Dataset.from_tensor_slices((xs, ys))
            .map(self.preprocess)
            .shuffle(len(ys))
            .batch(128)
        )


def main():
    (x_train, y_train), (x_val, y_val) = keras.datasets.fashion_mnist.load_data()
    logger.info("training and testing data loaded")
    my_data = MyFashionMNISTData()
    logger.info("fmnist data object created")
    train_datasets = my_data.create_dataset(x_train, y_train)
    val_datasets = my_data.create_dataset(x_val, y_val)

    my_dnn = MyDeepNeuralNetwork()
    my_dnn.train_model(train_datasets, val_datasets, plot_accuracy=True)


if __name__ == "__main__":
    main()
