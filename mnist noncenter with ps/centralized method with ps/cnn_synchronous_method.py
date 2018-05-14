from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import init_tower as tower
import time
import numpy as np
import argparse

NUM_WORKERS = 30
NUM_PS = 1
BATCH_SIZE = 2


def main():
    port = 24454
    cluster = tf.train.ClusterSpec({
        'ps': ['localhost:%d' % port],
        'worker': ['localhost:%d' % (port + i + 1) for i in range(NUM_WORKERS)]
    })
    if FLAGS.job_name == 'ps':
        with tf.device('/job:ps/task:0/cpu:0'):
            server = tf.train.Server(cluster, job_name='ps', task_index=FLAGS.task_index)
            server.join()

    else:
        is_chief = (FLAGS.task_index == 0)
        server = tf.train.Server(cluster, job_name='worker',
                                 task_index=FLAGS.task_index)

        worker_device = '/job:worker/task:%d/cpu:0' % FLAGS.task_index

        with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                                      cluster=cluster)):
            global_step = tf.train.get_or_create_global_step()
            x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
            y = tf.placeholder(tf.float32, [None, 10], name='y')
            logits = tower.tower(x)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)

            opt = tf.train.SyncReplicasOptimizer(
                tf.train.GradientDescentOptimizer(0.01),
                replicas_to_aggregate=NUM_WORKERS,
                total_num_replicas=NUM_WORKERS)

            merged = tf.summary.merge_all()
            sync_replicas_hook = opt.make_session_run_hook(is_chief, num_tokens=0)
            train_op = opt.minimize(loss, global_step=global_step)
            stop_hook = tf.train.StopAtStepHook(last_step=1001)

            # summary_hook = tf.train.SummarySaverHook(save_steps=10, output_dir=log_dir, summary_op=merged)
            with tf.train.MonitoredTrainingSession(master=server.target,
                                                   hooks=[sync_replicas_hook, stop_hook]) as sess:
                f = open('./cen_logdir_%s_%s.txt' % (FLAGS.job_name, FLAGS.task_index), 'w')
                start_time = time.time()
                for i in range(10000):
                    if sess.should_stop():
                        end_time = time.time()
                        print('time is', end_time - start_time)
                        f.close()
                        test_x, test_y = mnist.test.next_batch(batch_size=10000)
                        test_n = np.reshape(test_x, [-1, 28, 28, 1])
                        lo, accu = sess.run([loss, accuracy], feed_dict={x: test_n, y: test_y})
                        print("\n\nloss of test dataset is: ", lo)
                        print("accuracy of test dataset is: %.3f\n\n" % accu)
                        break
                    else:
                        batch_x, batch_y = mnist.train.next_batch(batch_size=BATCH_SIZE)
                        batch_n = np.reshape(batch_x, [-1, 28, 28, 1])
                        loss_value, _ = sess.run([loss, train_op],
                                                 feed_dict={x: batch_n, y: batch_y})
                        if i%200==0:
                            f.write(str(loss_value)+' ')
                            print('step is %d, tower_%d, loss is: %.4f' % (i, FLAGS.task_index, loss_value))

                        if i == 999:
                            end_time = time.time()
                            print('time is', end_time - start_time)
                            test_x, test_y = mnist.test.next_batch(batch_size=10000)
                            test_n = np.reshape(test_x, [-1, 28, 28, 1])
                            lo, accu = sess.run([loss, accuracy], feed_dict={x: test_n, y: test_y})
                            print("\n\nloss of test dataset is: ", lo)
                            print("accuracy of test dataset is: %.3f\n\n" % accu)


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
