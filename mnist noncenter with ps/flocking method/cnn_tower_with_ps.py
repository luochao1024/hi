"""the mnist convolutional tower"""
import tensorflow as tf
import numpy as np
import random
import csv

from flocking_optimizer import SHARED_VARIABLES_COLLECTION

w1 = np.array([0.0] * 5 * 5 * 32)
w2 = np.array([0.0] * 5 * 5 * 32 * 64)
w3 = np.array([0.0] * 7 * 7 * 64 * 100)
w4 = np.array([0.0] * 100 * 10)
ws = [w1, w2, w3, w4]
with open("init_same_variables.txt", 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    l = [element for element in lines]

    for i in range(5 * 5 * 32):
        w1[i] = float(l[0][i])

    for i in range(5 * 5 * 32 * 64):
        w2[i] = float(l[1][i])

    for i in range(7 * 7 * 64 * 100):
        w3[i] = float(l[2][i])

    for i in range(100 * 10):
        w4[i] = float(l[3][i])

w1_shaped = np.reshape(w1, [5, 5, 1, 32])
w2_shaped = np.reshape(w2, [5, 5, 32, 64])
w3_shaped = np.reshape(w3, [7 * 7 * 64, 100])
w4_shaped = np.reshape(w4, [100, 10])

init1 = tf.constant_initializer(w1_shaped)
init2 = tf.constant_initializer(w2_shaped)
init3 = tf.constant_initializer(w3_shaped)
init4 = tf.constant_initializer(w4_shaped)


def tower(images, flocking_towers, tower_index, d):

    random.seed(tower_index)
    w1 = np.array([0.0] * 5 * 5 * 32)
    w2 = np.array([0.0] * 5 * 5 * 32 * 64)
    w3 = np.array([0.0] * 7 * 7 * 64 * 100)
    w4 = np.array([0.0] * 100 * 10)
    ws = [w1, w2, w3, w4]
    with open("init_same_variables.txt", 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        l = [element for element in lines]

        for i in range(5 * 5 * 32):
            w1[i] = float(l[0][i]) + random.uniform(-d, d)

        for i in range(5 * 5 * 32 * 64):
            w2[i] = float(l[1][i]) + random.uniform(-d, d)

        for i in range(7 * 7 * 64 * 100):
            w3[i] = float(l[2][i]) + random.uniform(-d, d)

        for i in range(100 * 10):
            w4[i] = float(l[3][i]) + random.uniform(-d, d)

    w1_shaped = np.reshape(w1, [5, 5, 1, 32])
    w2_shaped = np.reshape(w2, [5, 5, 32, 64])
    w3_shaped = np.reshape(w3, [7 * 7 * 64, 100])
    w4_shaped = np.reshape(w4, [100, 10])

    init1 = tf.constant_initializer(w1_shaped)
    init2 = tf.constant_initializer(w2_shaped)
    init3 = tf.constant_initializer(w3_shaped)
    init4 = tf.constant_initializer(w4_shaped)

    # conv1
    # the tower_index is at the end of flocking_towers, so that the weights and biases used latter is
    # the ones placed on the local tower
    for index in flocking_towers:
        with tf.variable_scope('tower%d_conv1' % index):
            with tf.device('/job:ps/task:0'):
                if index == tower_index:
                    trainable = True
                else:
                    trainable = False

                weights = tf.get_variable('weights',
                                          shape=[5, 5, 1, 32],
                                          initializer=init1,
                                          trainable=trainable,
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                       '%s_tower_%d' % (SHARED_VARIABLES_COLLECTION, index)])
                biases = tf.get_variable('biases', shape=[32], initializer=tf.constant_initializer(0.0),
                                         trainable=trainable,
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                      '%s_tower_%d' % (SHARED_VARIABLES_COLLECTION, index)])

    conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2
    for index in flocking_towers:
        with tf.variable_scope('tower%d_conv2' % index):
            with tf.device('/job:ps/task:0'):
                if index == tower_index:
                    trainable = True
                else:
                    trainable = False

                weights = tf.get_variable('weights',
                                          shape=[5, 5, 32, 64],
                                          initializer=init2,
                                          trainable=trainable,
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                       '%s_tower_%d' % (SHARED_VARIABLES_COLLECTION, index)])
                biases = tf.get_variable('biases', shape=[64], initializer=tf.constant_initializer(0.0),
                                         trainable=trainable,
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                      '%s_tower_%d' % (SHARED_VARIABLES_COLLECTION, index)])

    conv = tf.nn.conv2d(pool1, weights, [1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation)

    # pool2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    # fully connected layer
    for index in flocking_towers:
        with tf.variable_scope('tower%d_fully_conn' % index):
            with tf.device('/job:ps/task:0'):
                if index == tower_index:
                    trainable = True
                else:
                    trainable = False

                weights = tf.get_variable('weights',
                                          shape=[7 * 7 * 64, 100],
                                          initializer=init3,
                                          trainable=trainable,
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                       '%s_tower_%d' % (SHARED_VARIABLES_COLLECTION, index)])
                biases = tf.get_variable('biases', shape=[100], initializer=tf.constant_initializer(0.0),
                                         trainable=trainable,
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                      '%s_tower_%d' % (SHARED_VARIABLES_COLLECTION, index)])

    flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])
    pre_activation = tf.matmul(flattened, weights) + biases
    fully_conn = tf.nn.relu(pre_activation)
    # drop_out = tf.nn.dropout(fully_conn, keep_prob=0.5)

    # logits layer
    for index in flocking_towers:
        with tf.variable_scope('tower%d_logits' % index):
            with tf.device('/job:ps/task:0'):
                if index == tower_index:
                    trainable = True
                else:
                    trainable = False

                weights = tf.get_variable('weights',
                                          shape=[100, 10],
                                          initializer=init4,
                                          trainable=trainable,
                                          collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                       '%s_tower_%d' % (SHARED_VARIABLES_COLLECTION, index)])
                biases = tf.get_variable('biases', shape=[10], initializer=tf.constant_initializer(0.0),
                                         trainable=trainable,
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                      '%s_tower_%d' % (SHARED_VARIABLES_COLLECTION, index)])

    logits = tf.add(tf.matmul(fully_conn, weights), biases)

    return logits
