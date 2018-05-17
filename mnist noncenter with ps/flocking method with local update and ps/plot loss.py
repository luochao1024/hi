import numpy as np
from scipy import *
import matplotlib.pyplot as plt

step = [1, 30, 60, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 5000, 10000]
test_loss = [2.303, 2.291, 2.211, 0.725, 0.382, 0.262, 0.193, 0.085, 0.064, 0.050, 0.052, 0.064]
test_accuracy = [0.197, 0.247, 0.410, 0.794, 0.888, 0.924, 0.943, 0.976, 0.984, 0.986, 0.989, 0.985]

p30_flocking = polyfit(step, test_loss, 30)
p30_centralized = polyfit(step, test_accuracy, 30)


plt.ylim((0, 3))
plt.xlim((0, 10000))
plt.xlabel('time(seconds)')
plt.ylabel('loss')
plt.plot(x1, polyval(p30_flocking, x1), 'r')
plt.plot(x2, polyval(p30_centralized, x2), 'b')
plt.plot(x1, loss_flocking, 'r', alpha=0.4,)
plt.plot(x2, loss_centralized, 'b', alpha=0.4)
plt.legend(['flocking', 'centralized'])
plt.show()



x1 = np.linspace(0, 48, num_lines1)
x2 = np.linspace(0, 48, num_lines2)
print(x1)

# plt.ylim((0, 1))
# plt.xlim((0, 48))
# plt.xlabel('time(seconds)')
# plt.ylabel('accuracy')
# plt.plot(x1, accuracy_flocking)
# plt.plot(x2, accuracy_nonflocking)
# plt.legend(['flocking', 'nonflocking'])
# plt.show()

plt.ylim((0, 3))
plt.xlim((0, 48))
plt.xlabel('time(seconds)')
plt.ylabel('cost')
plt.plot(x1, cost_flocking)
plt.plot(x2, cost_nonflocking)
plt.legend(['flocking', 'nonflocking'])
plt.show()
