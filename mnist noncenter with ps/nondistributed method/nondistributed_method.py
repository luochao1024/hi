from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import init_tower as tower
import time
import numpy as np
import argparse

BATCH_SIZE = 32

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

global_step = tf.train.get_or_create_global_step()
x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='y')
logits = tower.tower(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
opt = tf.train.GradientDescentOptimizer(0.01)

# opt = tf.train.MomentumOptimizer(0.01, momentum=0.99)
train_op = opt.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    start = time.time()
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
        batch_n = np.reshape(batch_x, [-1, 28, 28, 1])
        loss_value, _ = sess.run([loss, train_op],
                                 feed_dict={x: batch_n, y: batch_y})
        if i%200==0:
            print('step_%d, loss is %.3f' % (i, loss_value))
    end = time.time()
    test_x, test_y = mnist.test.next_batch(batch_size=10000)
    test_n = np.reshape(test_x, [-1, 28, 28, 1])
    lo, accu = sess.run([loss, accuracy], feed_dict={x: test_n, y: test_y})
    print("\n\nloss of test dataset is: ", lo)
    print("accuracy of test dataset is: %.3f\n\n" % accu)
    print('time is %.3f' % (end-start))