import numpy as np
import os
import urllib
import gzip
import struct
import mxnet as mx
import logging
logging.getLogger().setLevel(logging.DEBUG)


def read_data(label_url, image_url):
    with gzip.open(label_url) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_url, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


def mlp_relu_generator():

    # Create a place holder variable for the input data
    data = mx.sym.Variable('data')
    # Flatten the data from 4-D shape (batch_size, num_channel, width, height)
    # into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.Flatten(data=data)

    # The first fully-connected layer
    fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

    # The second fully-connected layer and the according activation function
    fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
    act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

    # The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
    fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
    out  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

    return out

def mlp_leaky_relu_generator():
    # Create a place holder variable for the input data
    data = mx.sym.Variable('data')
    # Flatten the data from 4-D shape (batch_size, num_channel, width, height)
    # into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.Flatten(data=data)

    # The first fully-connected layer
    fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.sym.LeakyReLU(data=fc1, name='relu1', act_type="leaky")

    # The second fully-connected layer and the according activation function
    fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
    act2 = mx.sym.LeakyReLU(data=fc2, name='relu2', act_type="leaky")

    # The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
    fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
    out  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    return out

def mlp_batch_norm_generator():
    # Create a place holder variable for the input data
    data = mx.sym.Variable('data')
    # Flatten the data from 4-D shape (batch_size, num_channel, width, height)
    # into 2-D (batch_size, num_channel*width*height)
    data = mx.sym.Flatten(data=data)

    # The first fully-connected layer
    fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
    norm1 = mx.sym.BatchNorm(fc1, fix_gamma=False)
    act1 = mx.sym.LeakyReLU(data=norm1, name='relu1', act_type="leaky")

    # The second fully-connected layer and the according activation function
    fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
    norm2 = mx.sym.BatchNorm(fc2, fix_gamma=False)
    act2 = mx.sym.LeakyReLU(data=norm2, name='relu2', act_type="leaky")

    # The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
    fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
    out  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
    return out

path='/home/ec2-user/MNIST_data/'
(train_lbl, train_img) = read_data(
    path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data(
    path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')


batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)
num_device = 1
devices = [mx.gpu(i) for i in range(num_device)]

# -----------LEAKY -------------------
mlp = mlp_relu_generator()
mlp_model = mx.model.FeedForward(
    symbol = mlp,       # network structure
    num_epoch = 10,     # number of data passes for training
    learning_rate = 0.1, # learning rate of SGD
    ctx = devices
)
mlp_model.fit(
    X=train_iter,       # training data
    eval_data=val_iter, # validation data
    batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
)

# -----------LEAKY -------------------
mlp_leaky = mlp_leaky_relu_generator()
mlp_leaky_model = mx.model.FeedForward(
    symbol = mlp_leaky,       # network structure
    num_epoch = 10,     # number of data passes for training
    learning_rate = 0.1, # learning rate of SGD
    ctx = devices
)
mlp_leaky_model.fit(
    X=train_iter,       # training data
    eval_data=val_iter, # validation data
    batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
)

# ----------- BATCH NORM --------------------
mlp_norm = mlp_batch_norm_generator()
mlp_norm_model = mx.model.FeedForward(
    symbol = mlp_norm,       # network structure
    num_epoch = 10,     # number of data passes for training
    learning_rate = 0.1, # learning rate of SGD
    ctx = devices
)
mlp_norm_model.fit(
    X=train_iter,       # training data
    eval_data=val_iter, # validation data
    batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
)

# --------- validate model accuracy with test-data iterator

mlp_acc = mlp_model.score(val_iter)
print 'Validation accuracy ReLU: %f%%' % (mlp_acc *100,)

mlp_leaky_acc = mlp_leaky_model.score(val_iter)
print 'Validation accuracy for LeakyReLU: %f%%' % (mlp_leaky_acc *100,)

mlp_norm_acc = mlp_norm_model.score(val_iter)
print 'Validation accuracy ReLu + BatchNorm: %f%%' % (mlp_norm_acc *100,)
