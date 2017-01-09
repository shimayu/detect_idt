import numpy as np
import argparse
import chainer 
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain, cuda
from chainer import report, training, datasets, iterators
from chainer.datasets import tuple_dataset
from chainer.training import extensions
from PIL import Image
import cv2

np.random.seed(1)

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

batchsize = 20
n_epoch = 200
n_units = 1000

class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            conv1 = L.Convolution2D(None, 20, 5),
            conv2 = L.Convolution2D(None, 50, 5),
            fc1 = L.Linear(None, 500),
            fc2 = L.Linear(None, 2),
            )
    def __call__(self, x, train=True):
        cv1 = self.conv1(x)
        relu = F.relu(cv1)
        h = F.max_pooling_2d(relu, 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.fc1(h)), train=train)
        return self.fc2(h)

model = L.Classifier(Model())
optimizer = optimizers.MomentumSGD(lr=0.001, momentum=0.9)
optimizer.setup(model)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    xp = cuda.cupy
else:
    xp = np

def conv(batch, batchsize):
    x, t = [], []
    for j in xrange(batchsize):
        x.append(batch[j][0])
        t.append(batch[j][1])   
    return Variable(xp.asarray(x)), Variable(xp.array(t))

dataset, targetset = [], []
data_train, data_test = [], []
target_train, target_test = [], []

Num_data = 400
Num_data_train_true = 160
Num_true = 200

path_1211 = "./Bitmap_12_11/"
path_plan = "PlanE_Next/"
path_devide = "devide_3/"
path_tr = "Bitmap_tr/"
path_b = "Bitmap_b/"

for i in xrange(Num_data_train_true):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_ideal_" + str(i) + ".bmp"
    dataset.append(cv2.imread(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    dataset.append(cv2.imread(filename))
for i in range(Num_data_train_true, Num_true):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_ideal_" + str(i) + ".bmp"
    dataset.append(cv2.imread(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    dataset.append(cv2.imread(filename))

# Set target number(0, 1) to targetset 
for i in range(0, Num_data_train_true):
    targetset.append(1)
    targetset.append(0)
for i in range(Num_data_train_true, Num_true):
    targetset.append(1)
    targetset.append(0)
# Convert List into ndarray
targetset = np.array(targetset).astype(np.int32)
dataset = np.array(dataset).astype(np.float32)
print(dataset.shape)
dataset = np.swapaxes(dataset, 1, 3)
print(dataset.shape)

# Split dataset to data_train(80%) and data_test(20%)
N = 320
data_train, data_test = np.split(dataset, [N])
target_train, target_test = np.split(targetset, [N])

# Learning loop
train = tuple_dataset.TupleDataset(data_train, target_train)
test = tuple_dataset.TupleDataset(data_test, target_test)

if __name__ == '__main__':
    for n in xrange(n_epoch):
        sum_loss = 0
        sum_accuracy = 0
        for i in chainer.iterators.SerialIterator(train, batchsize, 
                                                  repeat=False):
            x, t = conv(i, batchsize)
            model.zerograds()
            loss = model(x, t)
            loss.backward()
            optimizer.update()
            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
            
        print("train: loss = {0}, accuracy = {1}".format(sum_loss / N, sum_accuracy / N))

        # evaluation
        j = chainer.iterators.SerialIterator(test, batchsize).next()
        x, t = conv(j, batchsize)
        loss = model(x, t)
        print n, loss.data


# Save the model and the optimizer
print('save the model')
serializers.save_hdf5('mlp.model', model)
print('save the optimizer')
serializers.save_hdf5('mlp.state', optimizer)
