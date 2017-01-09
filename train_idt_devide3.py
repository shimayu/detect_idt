#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import six
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, serializers, computational_graph, cuda
import net
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

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

batchsize = 50
n_epoch = 150
n_units = 1000

# Prepare dataset

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

# Load all data to dataset
# for train
for i in range(0, 60):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_ideal_" + str(i) + ".bmp"
    dataset.append(Image.open(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    dataset.append(Image.open(filename))
for i in range(60, 110):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_up_" + str(i-60) + ".bmp"
    dataset.append(Image.open(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    dataset.append(Image.open(filename))
for i in range(110, 160):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_down_" + str(i-110) + ".bmp"
    dataset.append(Image.open(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    dataset.append(Image.open(filename))
# for test
for i in range(160, 180):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_ideal_" + str(i) + ".bmp"
    dataset.append(Image.open(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    dataset.append(Image.open(filename))
for i in range(180, 190):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_up_" + str(i-130) + ".bmp"
    dataset.append(Image.open(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    dataset.append(Image.open(filename))
for i in xrange(190, 200):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_down_" + str(i-140) + ".bmp"
    dataset.append(Image.open(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    dataset.append(Image.open(filename))

# Set target number(0, 1) to targetset 
for i in range(0, Num_data_train_true):
    targetset.append(1)
    targetset.append(0)
for i in range(Num_data_train_true, Num_true):
    targetset.append(1)
    targetset.append(0)

# Convert List into ndarray
for i in xrange(Num_data):
    dataset[i] = np.asarray(dataset[i]).astype(np.float32)
targetset = np.array(targetset).astype(np.int32)

# Split dataset to data_train(80%) and data_test(20%)
N = 320
data_train, data_test = np.split(dataset, [N])
target_train, target_test = np.split(targetset, [N])
N_test = target_test.size

# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'simple':    
    model = L.Classifier(net.MnistMLP(16128, n_units, 2))
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy
elif args.net == 'parallel':
    cuda.check_cuda_available()
    model = L.Classifier(net.MnistMLPParallel(16128, n_units, 2))
    xp = cuda.cupy

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_hdf5(args.resume, optimizer)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(data_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(target_train[perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        loss = model(x, t)
        loss.backward()
        optimizer.update(model, x, t)

        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('train mean loss={}, accuracy={}'.format(
        sum_loss / N, sum_accuracy / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(data_test[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(target_test[i:i + batchsize]),
                             volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test))

data_check, target_check = [], []
for i in range(200, 210):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_ideal_" + str(i) + ".bmp"
    data_check.append(Image.open(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    data_check.append(Image.open(filename))
for i in range(210, 215):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_up_" + str(i-150) + ".bmp"
    data_check.append(Image.open(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    data_check.append(Image.open(filename))
for i in range(215, 220):
    filename = path_1211 + path_plan + path_devide + path_tr + "input_down_" + str(i-155) + ".bmp"
    data_check.append(Image.open(filename))
    filename = path_1211 + path_plan + path_devide + path_b + "input_" + str(i) + ".bmp"
    data_check.append(Image.open(filename))

for i in xrange(20):
    data_check[i] = np.array(data_check[i]).astype(np.float32)
for i in xrange(20):
    x = chainer.Variable(xp.asarray([data_check[i]]), volatile='on')
    predict = F.softmax(model.predictor(x))
    print("predict[{0}]: {1}".format(i, predict.data))

# Save the model and the optimizer
print('save the model')
serializers.save_hdf5('mlp.model', model)
print('save the optimizer')
serializers.save_hdf5('mlp.state', optimizer)
