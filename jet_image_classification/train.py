from sklearn.utils import shuffle
import numpy as np
import mxnet as mx
from jet_resnet import jet_resnet


print("-----------Loading jet images------------------")
data_dir = '/home/ec2-user/data/jet_images/'
# data_dir = '/home/ec2-user/data/jet_images/'
data = np.load(data_dir + 'jetimages.npy')
print("-----------Done jet images------------------")

# Convert to images
X = data['image'].reshape((data.shape[0], 1, 25, 25)).astype('float32')
y = data['signal'].astype('int8')

# shuffle data
X, y = shuffle(X, y)


# split dataset
train_data = X[:50000, :].astype('float32')
train_label = y[:50000]
val_data = X[50000: 60000, :].astype('float32')
val_label = y[50000:60000]

# Normalize data
# train_data[:] /= 256.0
# val_data[:] /= 256.0
# create a numpy iterator
batch_size = 128
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, val_label, batch_size=batch_size)

num_device = 1
devices = [mx.gpu(i) for i in range(num_device)]

print("-----------Done arranging data------------------")

# ----------- RESNET --------------------
resnet = jet_resnet(2, 8, '1,25,25')
resnet_model = mx.model.FeedForward(
    symbol = resnet,       # network structure
    num_epoch = 100,     # number of data passes for training
    learning_rate = 0.05, # learning rate of SGD
    ctx = devices
)
print("-----------Start Training:------------------")
resnet_model.fit(
    X=train_iter,       # training data
    eval_data=val_iter, # validation data
    batch_end_callback = mx.callback.Speedometer(batch_size, 10) # output progress for each 200 data batches
)

print("-----------Done Training:------------------")
acc = resnet_model.score(val_iter)
print 'Validation accuracy: %f%%' % (acc *100,)









#
# import matplotlib.pyplot as plt
# import numpy as np
# import mxnet as mx
# import skimage
# import skimage.io as io
# import os
# %matplotlib inline
#
# data_dir = '/Users/mrestrep/Dropbox/BrownAgain/Data/jet_images/'
# # data_dir = '/home/ec2-user/data/jet_images/'
# data = np.load(data_dir + 'jetimages.npy')
#
# # Convert to images
# X = data['image'].reshape((data.shape[0], 1, 25, 25)).astype('float32')
# y = data['signal'].astype('int8')
# print(data['image'].shape)


# data_iter = mx.io.NDArrayIter(data=X,
#     label={'label': y},
#     batch_size=10)

# print(data_iter.provide_label)

# max_val
# # Normalize
# X = X.reshape(X.shape[0], -1)
# X = (X / np.sqrt((X ** 2).sum(-1))[:, None]).reshape((X.shape[0], 1, 25, 25))

# Explore
# plt.imshow((X_signal[1].reshape((25,25))), cmap='gray')
# X_signal = X[y==1]
# X_signal.shape
# io.find_available_plugins()
# io.use_plugin('pil', 'imsave')
#
#
# img_dir = data_dir + "imgs/signal/"
#
# if  not os.path.isdir(img_dir):
#     os.makedirs(img_dir)
#
#
# for i in range(0, 3):
#     fname = img_dir + "signal_" + str(i) + ".png"
#     img = X_signal[i].reshape((25,25))
#     max_val = np.amax(img)
#     min_val = np.amin(img)
#     img_norm = (img - min_val) * (2.0)/(max_val - min_val) - 1
#     print(np.amax(img_norm))
#
#     io.imsave(fname, skimage.img_as_float(img_norm))
