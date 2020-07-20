import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import os


output_dir = 'outputs'
ckpt_dir_train = os.path.join(output_dir, 'ckpt/train')
ckpt_dir_dev = os.path.join(output_dir, 'ckpt/dev')
log_dir = os.path.join(output_dir, 'log')


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


def define_ckpt(model, val_accuracy, optimizer, step, ckpt_dir_train, ckpt_dir_dev):
    ckpt = tf.train.Checkpoint(model=model, val_accuracy=val_accuracy, optimizer=optimizer, step=step)
    train_manager = tf.train.CheckpointManager(ckpt, ckpt_dir_train, max_to_keep=3)
    dev_manager = tf.train.CheckpointManager(ckpt, ckpt_dir_dev, max_to_keep=3)
    ckpt.restore(train_manager.latest_checkpoint)
    if train_manager.latest_checkpoint:
        print(f"Restored from {train_manager.latest_checkpoint}")
    else:
        print("Initializing from scratch.")
    return train_manager, dev_manager


def train():
    # 1. define model
    n_classes = 10
    model = define_model(n_classes)

    # 2. load data
    (train_x, train_y), (test_x, test_y) = load_data()

    # 3. define loss, optimizer, ckpt, val_accuracy
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    step = tf.Variable(1, name='global_step')
    optimizer = tf.optimizers.Adam(1e-3)
    val_accuracy = tf.Variable(0.0, name='val_accuracy', dtype=tf.float32)
    train_manager, val_manager = define_ckpt(model, val_accuracy, optimizer, step, ckpt_dir_train=ckpt_dir_train, ckpt_dir_dev=ckpt_dir_dev)

    print(f'@zkl start from step {step.numpy()} with val_accuracy {val_accuracy.numpy()}')

    # 4. define metrics
    accuracy = tf.metrics.Accuracy()
    mean_loss = tf.metrics.Mean(name='loss')

    # 5. define train_step
    @tf.function
    def train_step(inputs, labels, is_training=True):
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = loss(labels, logits)

        accuracy.update_state(labels, tf.argmax(logits, axis=-1))

        if is_training:
            gradients = tape.gradient(loss_value, model.trainable_variables)
            gradients = [tf.clip_by_norm(grad, 5) for grad in gradients]

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            step.assign_add(1)

        return loss_value, accuracy.result()

    # 6. define training configs
    epochs = 10
    batch_size = 32
    nr_batches_train = int(train_x.shape[0] / batch_size)
    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {nr_batches_train}")

    # 7. start real training
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    with train_summary_writer.as_default():
        for epoch in range(epochs):
            tf.random.set_seed(step.numpy())
            train_x = tf.random.shuffle(train_x)
            tf.random.set_seed(step.numpy())
            train_y = tf.random.shuffle(train_y)

            accuracy.reset_states()
            mean_loss.reset_states()

            for t in range(nr_batches_train):
                start_from = t * batch_size
                to = (t + 1) * batch_size
                features, labels = train_x[start_from:to], train_y[start_from:to]
                loss_value, accuracy_value = train_step(features, labels)

                mean_loss.update_state(loss_value)

                if t % 1000 == 0:
                    print(f"{step.numpy()}: {loss_value} - accuracy: {accuracy_value}")
                    save_path = train_manager.save()
                    print(f"Checkpoint saved: {save_path}")

                    tf.summary.image('train_set', features, max_outputs=3, step=step.numpy())
                    tf.summary.scalar('accuracy', accuracy_value, step=step.numpy())
                    tf.summary.scalar('loss', mean_loss.result(), step=step.numpy())

                    accuracy.reset_states()
                    mean_loss.reset_states()

            print(f"Epoch {epoch} terminated")

            accuracy.reset_states()
            mean_loss.reset_states()
            # Measuring accuracy on the whole testing set at the end of the epoch
            for t in range(int(test_x.shape[0] / batch_size)):
                start_from = t * batch_size
                to = (t + 1) * batch_size
                features, labels = test_x[start_from:to], test_y[start_from:to]
                loss_value, accuracy_value = train_step(features, labels, is_training=False)
                mean_loss.update_state(loss_value)

                if t % 1000 == 0:
                    tf.summary.scalar('val_accuracy', accuracy_value, step=step.numpy())
                    tf.summary.scalar('val_loss', mean_loss.result(), step=step.numpy())

                    accuracy.reset_states()
                    mean_loss.reset_states()

            history_val_accuracy = accuracy.result()
            print(f"Testing accuracy: {history_val_accuracy}")
            if history_val_accuracy > val_accuracy.numpy():
                val_accuracy.assign(history_val_accuracy)
                save_path = val_manager.save()
                print(f"Val Checkpoint saved: {save_path} with accuracy {val_accuracy.numpy()}")


train()
