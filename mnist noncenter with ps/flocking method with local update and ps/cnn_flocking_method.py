from __future__ import print_function

import argparse
import time
import random

import numpy as np
import tensorflow as tf
from flocking_optimizer import FlockingOptimizer
from tensorflow.examples.tutorials.mnist import input_data
import cnn_tower_with_ps as cnn_tower

BATCH_SIZE = 2
NUM_WORKERS = 45
NUM_FLOCKING_WORKERS = 3
ATTRACTION = 0.5
REPULSION = 0


def main():
    port = 9000
    log_dir = './flocking_noncenter_%.3f_%.3f_%s_r0.01' % (ATTRACTION, REPULSION, FLAGS.task_index)
    cluster = tf.train.ClusterSpec({
        'ps': ['localhost:%d' % port],
        'worker': ['localhost:%d' % (port + i + 1) for i in range(NUM_WORKERS)]
    })
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    if FLAGS.job_name == 'ps':
        with tf.device('/job:ps/task:0'):
            server = tf.train.Server(cluster, job_name='ps', task_index=FLAGS.task_index)
            server.join()

    else:
        server = tf.train.Server(cluster, job_name='worker',
                                 task_index=FLAGS.task_index)
        choices = [i for i in range(NUM_WORKERS) if i != FLAGS.task_index]
        print('this is in worker %d' % FLAGS.task_index, choices)
        flocking_workers = random.sample(choices, NUM_FLOCKING_WORKERS)
        flocking_workers.append(FLAGS.task_index)
        flocking_workers = tuple(flocking_workers)
        print('this is in worker %d' % FLAGS.task_index, flocking_workers)
        with tf.device('/job:ps/task:0'):
            global_step = tf.Variable(0, name='global_step', trainable=False)

        worker_device = '/job:worker/task:%d' % FLAGS.task_index
        with tf.device(worker_device):
            x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
            y = tf.placeholder(tf.float32, [None, 10], name='y')
            logits = cnn_tower.tower(x, flocking_workers, FLAGS.task_index)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # tf.summary.scalar('loss', loss)
            # tf.summary.scalar('accuracy', accuracy)

            sgd_opt = tf.train.MomentumOptimizer(0.01, momentum=0.99)
            # sgd_opt = tf.train.GradientDescentOptimizer(0.01)
            opt = FlockingOptimizer(opt=sgd_opt,
                                    attraction=ATTRACTION,
                                    repulsion=REPULSION)


            # scaffold = tf.train.Scaffold(init_op=init)
            # merged = tf.summary.merge_all()
            train_op = opt.minimize_with_flocking(loss=loss, flocking_workers=flocking_workers)
            # stop_hook = tf.train.StopAtStepHook(last_step=5000)
            # summary_hook = tf.train.SummarySaverHook(save_steps=10, output_dir=log_dir, summary_op=merged)
            init = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            print('this is local variables', tf.global_variables())

            f = open('./flocking_noncenter_%.3f_%.3f_%s_r0.01.txt' % (ATTRACTION, REPULSION, FLAGS.task_index), 'w')
            with tf.Session(target=server.target) as sess:
                sess.run([init, init_local])
                start_time = time.time()
                for i in range(1001):
                    batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
                    batch_n = np.reshape(batch_x, [-1, 28, 28, 1])
                    _, loss_value = sess.run([train_op, loss], feed_dict={x: batch_n, y: batch_y})
                    if i%200 == 0:
                        f.write(str(loss_value) + ' ')
                        print('step is %d, tower_%d, loss is %.3f' % (i, FLAGS.task_index, loss_value))
                end_time = time.time()
                print('this is worker_%d, time is' % FLAGS.task_index, end_time - start_time)
                print('num in tf.trainable_variables is', len(tf.trainable_variables()))
                f.close()
                test_x, test_y = mnist.test.next_batch(batch_size=10000)
                test_n = np.reshape(test_x, [-1, 28, 28, 1])
                lo, accu = sess.run([loss, accuracy], feed_dict={x:test_n, y:test_y})
                print("\n\nthis is worker_%d loss of test dataset is: %.3f"%(FLAGS.task_index,lo))
                print("this is worker_%d, accuracy of test dataset is: %.3f\n\n" % (FLAGS.task_index, accu))
                server.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec

    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
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
