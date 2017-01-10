import numpy as np
import sys
import matplotlib.pyplot as plt

data = []
accuracy_train, accuracy_test = [], []
loss_train, loss_test = [], []
epoch = []

data = np.loadtxt(sys.argv[1], delimiter=',', dtype='float')

for i in xrange(len(data)):
    if i % 4 == 0:
        loss_train.append(data[i])
    elif i % 4 == 1:
        accuracy_train.append(data[i])
    elif i % 4 == 2:
        loss_test.append(data[i])
    elif i % 4 == 3:
        accuracy_test.append(data[i])

for i in xrange(len(loss_train)):
    print("loss_train{0}: {1}".format(i, loss_train[i]))

for i in xrange(len(accuracy_train)):
    print("accuracy_train{0}: {1}".format(i, accuracy_train[i]))

for i in xrange(len(loss_test)):
    print("loss_test{0}: {1}".format(i, loss_test[i]))

for i in xrange(len(accuracy_test)):
    print("accuracy_test{0}: {1}".format(i, accuracy_test[i]))


for i in xrange(len(accuracy_train)):
    epoch.append(i)

plt.plot(epoch, accuracy_train, label = "accuracy_train")
plt.plot(epoch, accuracy_test, label = "accuracy_test")
plt.legend(loc='lower right')
plt.title("accuracy_train and accuracy_test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.xlim(-1, 155)
plt.ylim(0, 1.05)
# plt.show()
plt.savefig("accuracy.png")
plt.clf()

plt.plot(epoch, loss_train, label = "loss_train")
plt.plot(epoch, loss_test, label = "loss_test")
plt.legend()
plt.title("loss_train and loss_test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.xlim(0, 155)
plt.ylim(-5, 1000)
# plt.show()
plt.savefig("loss.png")
