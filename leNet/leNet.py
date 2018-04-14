import tensorflow as tf
import numpy as np

def leNetModel(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1,28,28,1])

    # Conv Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=(5, 5),
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=(2,2),
        strides=2
    )

    # Conv Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=(5,5),
        padding="same",
        activation=tf.nn.relu
    )

    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=(2,2),
        strides=2
    )

    # Dense Layer 1
    pool2_flatten = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense1 = tf.layers.dense(
        inputs=pool2_flatten,
        units=1024,
        activation=tf.nn.relu
    )
    dropout1 = tf.layers.dropout(
        inputs=dense1,
        rate=0.4,
        training=mode==tf.estimator.ModeKeys.TRAIN
    )

    # Logits Layer
    logits = tf.layers.dense(
        inputs=dropout1,
        units=10,
    )

    prediction = {
        "classes" : tf.argmax(logits, axis=1),
        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
        # "accuracy" : tf.constant(tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))[0],\
        #                          name="accuracy", dtype=tf.float32)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)

    # Calculate Loss and Accuracy (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=prediction["classes"],
                                   name='acc_op')
    eval_metric_ops = {"accuracy": accuracy}
    tf.summary.scalar("loss", loss)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=leNetModel, model_dir="./tmp/mnist_convnet_model"
    )

    # Set up logging for predictions
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=50)
    tf.logging.set_verbosity(tf.logging.INFO)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=50,
        num_epochs=1000,
        shuffle=True
    )

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=5000
        # hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    main()