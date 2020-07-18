import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import numpy as np


def define_cnn(x, n_classes, reuse, is_training):
    with tf.variable_scope('cnn', reuse=reuse):
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 63, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        shape = (-1, conv2.shape[1].value * conv2.shape[2].value * conv2.shape[3].value)
        fc1 = tf.reshape(conv2, shape)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=0.5, training=is_training)

        out = tf.layers.dense(fc1, n_classes)

    return out


def prepare_dataset():
    (train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

    # Scale input in [-1, 1] range
    train_x = train_x / 255. * 2 - 1
    test_x = test_x / 255. * 2 - 1

    # Add the last 1 dimension, so to have images 28x28x1
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)

    return train_x, train_y, test_x, test_y


def train():
    input = tf.placeholder(tf.float32, (None, 28, 28, 1))
    labels = tf.placeholder(tf.int64, (None, ))

    logits = define_cnn(input, 10, reuse=False, is_training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    global_step = tf.train.get_or_create_global_step()
    train_op = tf.train.AdamOptimizer().minimize(loss, global_step)

    writer = tf.summary.FileWriter('log/graph_loss', tf.get_default_graph())
    validation_summary_writer = tf.summary.FileWriter('log/graph_loss/validation')

    predictions = tf.argmax(logits, axis=1)
    correct_predictions = tf.equal(predictions, labels)
    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32), name='accuracy'
    )
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    loss_summary = tf.summary.scalar('loss', loss)

    train_x, train_y, test_x, test_y = prepare_dataset()

    epochs = 10
    batch_size = 32
    nr_batches_train = int(train_x.shape[0] / batch_size)

    print(f"Batch size: {batch_size}")
    print(f"Number of batches per epoch: {nr_batches_train}")

    validation_accuracy = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            for t in range(nr_batches_train):
                start_from = t * batch_size
                to = (t + 1) * batch_size

                loss_value, _, step = sess.run([loss, train_op, global_step], feed_dict={
                    input: train_x[start_from: to],
                    labels: train_y[start_from: to]
                })

                if t % 10 == 0:
                    print(f"{step}: {loss_value}")

            print(f"Epoch {epoch} terminated: measuring metrics and logging summaries")

            saver.save(sess, 'log/graph_loss/model')

            start_from = 0
            to = 128
            train_accuracy_summary, train_loss_summary = sess.run(
                [accuracy_summary, loss_summary],
                feed_dict={
                    input: train_x[start_from: to],
                    labels: train_y[start_from: to]
                }
            )

            validation_accuracy_summary, validation_accuracy_value, validation_loss_summary = sess.run(
                [accuracy_summary, accuracy, loss_summary],
                feed_dict={
                    input: test_x[start_from: to],
                    labels: test_y[start_from: to]
                }
            )

            # save values in TensorBoard
            writer.add_summary(train_accuracy_summary, step)
            writer.add_summary(train_loss_summary, step)
            validation_summary_writer.add_summary(validation_accuracy_summary, step)
            validation_summary_writer.add_summary(validation_loss_summary, step)

            writer.flush()
            validation_summary_writer.flush()

            # model selection
            if validation_accuracy_value > validation_accuracy:
                print(f'@zkl: a higher validation accuray {validation_accuracy_value} at epoch {epoch}')
                validation_accuracy = validation_accuracy_value
                saver.save(sess, 'log/graph_loss/best_model/best')

    writer.close()


train()
