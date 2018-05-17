import numpy as np
from scipy import *
import matplotlib.pyplot as plt

step = [1, 30, 60, 100, 200, 300, 400, 450, 500, 600, 700, 800, 1000, 1500, 2000, 3000, 5000]
test_loss____ =   [2.303, 2.296, 2.295, 2.292, 2.285, 2.213, 2.278, 2.012, 1.446, 1.487, 0.900, 0.339, 0.240, 0.114, 0.086, 0.059, 0.055]
test_accuracy =   [0.175, 0.190, 0.195, 0.286, 0.103, 0.213, 0.104, 0.427, 0.627, 0.597, 0.785, 0.921, 0.941, 0.970, 0.976, 0.982, 0.985]

#centralized method
centr_step = [1, 30, 60, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 5000]
centr_test_loss = [2.302, 2.290, 2.275, 2.248, 2.030, 0.913, 0.570, 0.399, 0.344, 0.345, 0.288, 0.250, 0.237, 0.190, 0.137, 0.107, 0.068]
centr_accuracy_ = [0.117, 0.155, 0.259, 0.293, 0.617, 0.790, 0.827, 0.882, 0.900, 0.894, 0.913, 0.925, 0.930, 0.943, 0.961, 0.970, 0.979]

m_step = [1, 30, 60, 100, 150, 200, 250, 500, 1000, 3000, 5000]
m_test_loss____ = [2.303, 2.291, 2.211, 0.725, 0.382, 0.262, 0.193, 0.085, 0.064, 0.050, 0.052]
m_test_accuracy = [0.197, 0.247, 0.410, 0.794, 0.888, 0.924, 0.943, 0.976, 0.984, 0.986, 0.989]


# p40_test_loss = polyfit(step, test_loss____, 11)
# p40_test_accuracy = polyfit(step, test_accuracy, 11)

# p40_m_test_loss = polyfit(m_step, m_test_loss____, 6)
# p40_m_test_accuracy = polyfit(m_step, m_test_accuracy, 6)


plt.ylim((0, 3))
plt.xlim((0, 5000))
plt.xlabel('time(seconds)')
plt.ylabel('loss')
plt.plot(m_step, m_test_loss____, 'r')
plt.plot(centr_step, centr_test_loss)
# plt.plot(m_step, polyval(p40_m_test_loss, m_step), 'r')
plt.plot(step, test_loss____)
# plt.plot(step, polyval(test_loss____, step), 'b')
# plt.plot(x1, loss_flocking, 'r', alpha=0.4,)
# plt.plot(x2, loss_centralized, 'b', alpha=0.4)
plt.legend(['flocking', 'centralized'])
plt.show()



# plt.plot(m_step, polyval(p40_m_test_accuracy, m_step), 'b')


# x1 = np.linspace(0, 48, num_lines1)
# x2 = np.linspace(0, 48, num_lines2)
# print(x1)

# plt.ylim((0, 1))
# plt.xlim((0, 48))
# plt.xlabel('time(seconds)')
# plt.ylabel('accuracy')
# plt.plot(x1, accuracy_flocking)
# plt.plot(x2, accuracy_nonflocking)
# plt.legend(['flocking', 'nonflocking'])
# plt.show()

# plt.ylim((0, 3))
# plt.xlim((0, 48))
# plt.xlabel('time(seconds)')
# plt.ylabel('cost')
# plt.plot(x1, cost_flocking)
# plt.plot(x2, cost_nonflocking)
# plt.legend(['flocking', 'nonflocking'])
# plt.show()
