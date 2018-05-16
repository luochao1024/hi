from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cifar10_flocking as cifar10
import time
from datetime import datetime
import random
import argparse
from flocking_optimizer_cifar10 import \
    FlockingOptimizer, FlockingCustomGetter, GLOBAL_VARIABLE_NAME, \
    LOCAL_VARIABLE_NAME, RECORD_AVERAGE_VARIABLE_NAME

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '',
                           """One of 'ps', 'worker' """)
tf.app.flags.DEFINE_integer('task_index', 0,
                            """Index of task within the job""")
tf.app.flags.DEFINE_string('train_dir', './multi_gpus_cifar10_flocking',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

NUM_WORKERS = 3
NUM_FLOCKING_WORKERS = 1
ATTRACTION = 0.5
REPULSION = 3.0


def train():
    port = 15176
    log_dir = './flocking_noncenter_%.3f_%.3f_%s_%s_r0.1' % (ATTRACTION, REPULSION, FLAGS.job_name, FLAGS.task_index)
    cluster = tf.train.ClusterSpec({
        'ps': ['localhost:%d' % port],
        'worker': ['localhost:%d' % (port+i+1) for i in range(NUM_WORKERS)]
    })
    if FLAGS.job_name == 'ps':
        with tf.device('/job:ps/task:0/cpu:0'):
            server = tf.train.Server(cluster, job_name='ps', task_index=FLAGS.task_index)
            server.join()

    else:
        choices = [i for i in range(NUM_WORKERS) if i != FLAGS.task_index]
        print('this is in worker %d' % FLAGS.task_index, choices)
        flocking_workers = random.sample(choices, NUM_FLOCKING_WORKERS)
        flocking_workers.append(FLAGS.task_index)
        flocking_workers = tuple(flocking_workers)
        print('this is in worker %d' % FLAGS.task_index, flocking_workers)

        # is_chief = (FLAGS.task_index == 0)
        # gpu_options = tf.GPUOptions(allow_growth=True,
        #                             allocator_type="BFC", visible_device_list="%d" % FLAGS.task_index)
        # config = tf.ConfigProto(gpu_options=gpu_options,
        #                         allow_soft_placement=True)
        server = tf.train.Server(cluster, job_name='worker',
                                 task_index=FLAGS.task_index,)

        worker_device = '/job:worker/task:%d' % FLAGS.task_index
        # f_getter = FlockingCustomGetter(flocking_workers=flocking_workers, worker_index=FLAGS.task_index)

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()

        with tf.device(worker_device):
            # Build a Graph that computes the logits predictions from the
            # inference model.
            logits = cifar10.inference(images, flocking_workers=flocking_workers)

            # Calculate loss.
            loss = cifar10.loss(logits, labels)

            # Build a Graph that trains the model with one batch of examples and
            # updates the model parameters.
            sgd_opt = tf.train.GradientDescentOptimizer(0.01)
            f_opt = FlockingOptimizer(
                opt=sgd_opt,
                flocking_workers=flocking_workers,
                attraction=ATTRACTION,
                repulsion=REPULSION)

            grads = f_opt.compute_gradients(loss)

            # Apply gradients.
            train_op = f_opt.apply_gradients_and_flocking(grads)

            # class _LoggerHook(tf.train.SessionRunHook):
            #     def begin(self):
            #         self._start_time = time.time()
            #         self._step = -1
            #
            #     def before_run(self, run_context):
            #         self._step += 1
            #         return tf.train.SessionRunArgs(loss)  # Asks for loss value.
            #
            #     def after_run(self, run_context, run_values):
            #         if self._step % FLAGS.log_frequency == 0:
            #             current_time = time.time()
            #             duration = current_time - self._start_time
            #             self._start_time = current_time
            #
            #             loss_value = run_values.results
            #             examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
            #             sec_per_batch = float(duration / FLAGS.log_frequency)
            #
            #             format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
            #                           'sec/batch)')
            #             print(format_str % (datetime.now(), self._step, loss_value,
            #                                 examples_per_sec, sec_per_batch))

            init = tf.global_variables_initializer()


        f = open('./flocking_noncenter_%.3f_%.3f_%s_%s_r0.01.txt' % (ATTRACTION, REPULSION, FLAGS.job_name, FLAGS.task_index), 'w')
        with tf.Session(target=server.target) as sess:
            start = time.time()
            sess.run(init)
            for step in range(1001):
                print(step)
                sess.run(train_op)
                # loss_value, _ = sess.run([loss, train_op])
                # if not step % FLAGS.log_frequency:
                #     print('step: %d, loss is %.2f' % (step, loss_value))
                #     f.write(str(loss_value) + ',')
            end = time.time()
            print('elapsed time is', end-start)
            print('attraction is', ATTRACTION, 'repulsion is', REPULSION)
        server.join()

def main(argv=None):  # pylint: disable=unused-argument
    FLAGS.train_dir = FLAGS.train_dir + str(FLAGS.task_index)
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
