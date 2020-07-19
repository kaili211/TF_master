import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def define_model(n_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 5, activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Conv2D(64, 3, activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes)
    ])

    model.summary()

    return model


def load_data():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()
    # Scale input in [-1, 1] range
    train_x = tf.expand_dims(train_x, -1)
    train_x = (tf.image.convert_image_dtype(train_x, tf.float32) - 0.5) * 2
    train_y = tf.expand_dims(train_y, -1)

    test_x = tf.expand_dims(test_x, -1)
    test_x = (tf.image.convert_image_dtype(test_x, tf.float32) - 0.5) * 2
    test_y = tf.expand_dims(test_y, -1)

    return (train_x, train_y), (test_x, test_y)


def train():
    pass


train()
