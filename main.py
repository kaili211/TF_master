import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def define_model():
    n_classes = 10
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def prepare_dataset():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    # Scale input in [-1, 1] range
    train_x = train_x / 255. * 2 - 1
    test_x = test_x / 255. * 2 - 1

    # Add the last 1 dimension, so to have images 28x28x1
    train_x = tf.expand_dims(train_x, -1).numpy()
    test_x = tf.expand_dims(test_x, -1).numpy()

    return train_x, train_y, test_x, test_y


def train():
    model = define_model()
    train_x, train_y, test_x, test_y = prepare_dataset()
    model.fit(train_x, train_y, epochs=10)
    model.evaluate(test_x, test_y)

train()
