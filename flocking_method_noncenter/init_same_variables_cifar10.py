import tensorflow as tf
import numpy as np
weights1 = tf.truncated_normal(shape=[5, 5, 3, 64],
                              stddev=5e-2)   
weights2 = tf.truncated_normal(shape=[5, 5, 64, 64],
                          stddev=5e-2) 
weights3 =tf.truncated_normal(shape=[6*6*64, 384],
                         stddev=0.04)   
weights4 = tf.truncated_normal(shape=[384, 192],
                          stddev=0.04) 
weights5 = tf.truncated_normal(shape=[192, 10],
                          stddev=1/192.0)

file_weights = open('init_same_variables.txt', 'w')
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	w1 = weights1.eval()
	w2 = weights2.eval()
	
	w3 = weights3.eval()
	w4 = weights4.eval()
	w5 = weights5.eval()
	l = [w1, w2, w3, w4, w5]
	for i in range(5):
		arr = np.reshape(l[i], [-1])
		print(len(arr))
		for ele in arr:
			file_weights.write(str(ele)+',')
		file_weights.write('\n')
