from __future__ import print_function

import argparse
import time
import random

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 4
NUM_TOWERS = 3
NUM_FLOCKING_TOWERS = 2
ATTRACTION = 0.9
REPULSION = 0.0


def main():
    port = 15176
    cluster = tf.train.ClusterSpec({'tower': ['localhost:%d' % (port + i) for i in range(NUM_TOWERS)]})

    server = tf.train.Server(cluster, job_name='tower',
                             task_index=FLAGS.task_index)

    with tf.device('/task:0'):
        x = tf.Variable(1, name='x')

    with tf.device('/task:1'):
        y = tf.Variable(1, name='y')

    with tf.device('/task:1'):
        z = tf.Variable(1, name='z')

    with tf.device('/task:%d'%FLAGS.task_index):
        new_x = tf.identity(x)
        new_y = tf.identity(y)
        new_z = tf.identity(z)
        c = new_x * new_y * new_z
        # c = x * y * z

    init = tf.global_variables_initializer()

    with tf.Session(target=server.target) as sess:
        sess.run(init)
        start_time = time.time()
        for i in range(1, 20001):
            c_value = sess.run(c)
            if i%5000 == 0:
                print('step is %d, tower_%d, loss is %.3f' % (i, FLAGS.task_index, c_value))
                print('worker %d, x device is'%FLAGS.task_index, x.device)
                print('worker %d, y device is'%FLAGS.task_index, y.device)
                print('worker %d, c device is'%FLAGS.task_index, c.device)
                # print('worker %d, new_x device is' % FLAGS.task_index, new_x.device)
                # print('worker %d, new_y device is' % FLAGS.task_index, new_y.device)

        end_time = time.time()
        print('time is', end_time - start_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec

    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    main()
