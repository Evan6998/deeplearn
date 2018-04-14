import tensorflow as tf
import numpy as np
import cifar10 

def vggModel(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1,32,32,3])

    conv1_1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=(7,7),
        padding="same",
        activation=tf.nn.relu
    )
    conv1_2 = tf.layers.conv2d(
        inputs=conv1_1,
        filters=64,
        kernel_size=(3,3),
        padding="same",
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1_2,
        pool_size=(2,2),
        strides=2
    )

    conv2_1 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=(3,3),
        padding="same",
        activation=tf.nn.relu
    )
    conv2_2 = tf.layers.conv2d(
        inputs=conv2_1,
        filters=128,
        kernel_size=(3,3),
        padding="same",
        activation=tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2_2,
        pool_size=(2,2),
        strides=2
    )

    conv3_1 = tf.layers.conv2d(
        inputs=pool2,
        filters=256,
        kernel_size=(3,3),
        padding="same",
        activation=tf.nn.relu 
    )
    conv3_2 = tf.layers.conv2d(
        inputs=conv3_1,
        filters=256,
        kernel_size=(3,3),
        padding="same",
        activation=tf.nn.relu 
    )
    conv3_3 = tf.layers.conv2d(
        inputs=conv3_2,
        filters=512,
        kernel_size=(3,3),
        padding="same",
        activation=tf.nn.relu 
    )

    pool3 = tf.layers.max_pooling2d(
        inputs=conv3_3,
        pool_size=(8,8),
        strides=2
    )

    pool_flatten = tf.reshape(pool3, [-1,1*1*512])
    dense1 = tf.layers.dense(
        inputs=pool_flatten,
        units=1024,
        activation=tf.nn.relu
    )
    dropout1 = tf.layers.dropout(
        inputs=dense1,
        training=mode==tf.estimator.ModeKeys.TRAIN
    )
    # dense2 = tf.layers.dense(
    #     inputs=dropout1,
    #     units=4096,
    #     activation=tf.nn.relu
    # )
    # dropout2 = tf.layers.dropout(
    #     inputs=dense2,
    #     training=mode==tf.estimator.ModeKeys.TRAIN
    # )
    logits = tf.layers.dense(
        inputs=dropout1,
        units=10
    )

    predictions = {
        "classes" : tf.argmax(logits, axis=1),
        "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions["classes"],
                                   name="acc_op")
    eval_metric_ops = {"accuracy":accuracy}
    tf.summary.scalar("loss", loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        # optimizer = tf.train.exponential_decay(0.001, tf.train.global_step(), 1000, 0.96, staircase=True)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)

    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)


def main():
    train_data, train_labels, eval_data, eval_labels = cifar10.load('./cifar-10-batches-py')
    cifar_classifier = tf.estimator.Estimator(
        model_fn=vggModel,model_dir="./tmp/cifar_vgg_model"
    )

    tf.logging.set_verbosity(tf.logging.INFO)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":train_data},
        y=train_labels,
        batch_size=64,
        num_epochs=30,
        shuffle=True
    )

    cifar_classifier.train(input_fn=train_input_fn, steps=30000)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y = eval_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_result = cifar_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_result)

if __name__ == '__main__':
    with tf.device('device:GPU:3') :
        main()