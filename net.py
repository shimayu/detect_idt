import chainer
import chainer.functions as F
import chainer.links as L


class MnistMLP(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    This is a very simple implementation of an MLP. You can modify this code to
    build your own neural net.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

class MyMnistMLP(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    This is a very simple implementation of an MLP. You can modify this code to
    build your own neural net.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MyMnistMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_units),
            l4=L.Linear(n_units, n_units),
            l5=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.l5(h4)

class Model(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(Model, self).__init__(
            conv1 = L.Convolution2D(None, 20, 5),
            conv2 = L.Convolution2D(None, 50, 5),
            fc1 = L.Linear(None, n_units),
            fc2 = L.Linear(None, n_out),
            )
    def __call__(self, x, train=True):
        cv1 = self.conv1(x)
        relu = F.relu(cv1)
        h = F.max_pooling_2d(relu, 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.fc1(h)), train=train)
        return self.fc2(h)

class MnistMLPParallel(chainer.Chain):

    """An example of model-parallel MLP.

    This chain combines four small MLPs on two different devices.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLPParallel, self).__init__(
            first0=MnistMLP(n_in, n_units // 2, n_units).to_gpu(0),
            first1=MnistMLP(n_in, n_units // 2, n_units).to_gpu(1),
            second0=MnistMLP(n_units, n_units // 2, n_out).to_gpu(0),
            second1=MnistMLP(n_units, n_units // 2, n_out).to_gpu(1),
        )

    def __call__(self, x):
        # assume x is on GPU 0
        x1 = F.copy(x, 1)

        z0 = self.first0(x)
        z1 = self.first1(x1)

        # sync
        h0 = z0 + F.copy(z1, 0)
        h1 = z1 + F.copy(z0, 1)

        y0 = self.second0(F.relu(h0))
        y1 = self.second1(F.relu(h1))

        # sync
        y = y0 + F.copy(y1, 0)
        return y