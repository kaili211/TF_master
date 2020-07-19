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
    n_classes = 10
    model = define_model(n_classes)

    (train_x, train_y), (test_x, test_y) = load_data()

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    step = tf.Variable(1, name='global_step')
    optimizer = tf.optimizers.Adam(1e-3)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")

    accuracy = tf.metrics.Accuracy()
    mean_loss = tf.metrics.Mean(name='loss')

    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = loss(labels, logits)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        gradients = [tf.clip_by_norm(grad, 5) for grad in gradients]

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        step.assign_add(1)

        accuracy.update_state(labels, tf.argmax(logits, axis=-1))
        return loss_value, accuracy.result()

    epochs = 10
    batch_size = 32
    nr_batches_train = int(train_x.shape[0] / batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {nr_batches_train}")

    train_summary_writer = tf.summary.create_file_writer('./log/train')
    with train_summary_writer.as_default():
        for epoch in range(epochs):
            for t in range(nr_batches_train):
                start_from = t * batch_size
                to = (t + 1) * batch_size
                features, labels = train_x[start_from:to], train_y[start_from:to]
                loss_value, accuracy_value = train_step(features, labels)

                mean_loss.update_state(loss_value)

                if t % 10 == 0:
                    print(f"{step.numpy()}: {loss_value} - accuracy: {accuracy_value}")
                    save_path = manager.save()
                    print(f"Checkpoint saved: {save_path}")

                    tf.summary.image('train_set', features, max_outputs=3, step=step.numpy())
                    tf.summary.scalar('accuracy', accuracy_value, step=step.numpy())
                    tf.summary.scalar('loss', mean_loss.result(), step=step.numpy())

                    accuracy.reset_states()
                    mean_loss.reset_states()

        print(f"Epoch {epoch} terminated")

        accuracy.reset_states()
        # Measuring accuracy on the whole training set at the end of the epoch
        for t in range(nr_batches_train):
            start_from = t * batch_size
            to = (t + 1) * batch_size
            features, labels = train_x[start_from:to], train_y[start_from:to]
            logits = model(features)
            accuracy.update_state(labels, tf.argmax(logits, -1))

        print(f"Training accuracy: {accuracy.result()}")
        accuracy.reset_states()


train()
