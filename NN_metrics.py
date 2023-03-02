# Three ways to check the success / failure of you neural network

import tensorflow as tf
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

##################################################################################################################################

# Check Loss Function (you want to see your loss decrease w/ epochs)
### Loss should be >0.6 (very dependent on problem)

# define inputs (likely defined much earlier)
nodes = 60
LR = 0.0001  # learning rate

# train network (see neural_net_architecture for more info)
history = model.fit(x_train_shp, hotlabels_train,
                    validation_data=(x_val_shp, hotlabels_val),
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,  # shuffle data before each epoch
                    verbose=0,
                    callbacks=callbacks)

# Grab accuracy and loss values from training
out = history.history
out_list = list(out.items())

# acc of validation
acc = out_list[5]
acc = np.array(acc[1]) * 100

# training loss
loss = out_list[0]
loss = loss[1]

# plot loss (full)
eppy = np.arange(1, len(loss)+1)
plt.title('Loss Function\nLR:{} - N:{}'.format(LR, nodes), fontsize=16)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.axhline(y = 0.6, color='gray', linestyle = '--', linewidth=2)  # below 0.6 is typically "good"
plt.plot(eppy, loss, linewidth=5)
plt.show()

# plot loss (zoomed in between 0 and 1)
plt.title('Loss Function\nLR:{} - N:{}'.format(LR, nodes), fontsize=18)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.axhline(y = 0.6, color='gray', linestyle = '--', linewidth=2)
plt.plot(eppy, loss, linewidth=1)
plt.ylim([0, 1])
plt.show()

##################################################################################################################################

# Check distrubition of confidences
### Should have some variety with a lot near 1.

# define bins
bin_num = 50

# Get model confidences (see neural_net_architecture for more info)
Conf_all = model.predict(x_val_shp)  # Confidences for all classes
WConf = np.argmax(Conf_all, axis=1)  # index of winning confidence
Conf = np.amax(Conf_all, 1)  # array of just the winning confidences

# plot distribution
plt.title('Distrubition of Confidences\nLR:{} - N:{}'.format(LR, nodes), fontsize=18)
plt.xlabel('Frequenct', fontsize=16)
plt.ylabel('Confidence', fontsize=16)
plt.hist(Conf, bin_num, density = 1, alpha = 0.7)
plt.show()

##################################################################################################################################

# Heat map for classifcation problem
### Heat map: What classes did the network predict compared to the actual class
### If the network is performing very well, we should expect large values across the diagonal

# specify number of classes
num_cls = 3

# Get model confidences (see neural_net_architecture for more info)
Conf_all = model.predict(x_val_shp)  # Confidences for all classes
WConf = np.argmax(Conf_all, axis=1)  # index of winning confidence
Conf = np.amax(Conf_all, 1)  # array of just the winning confidences

# put necessary info together
hit_miss1 = np.stack((Conf, y_val, WConf), axis=-1)  # y_val = class values - defined before running network
hit_miss = hit_miss1[hit_miss1[:,1].argsort()[::-1]]  # sorted from most to least confident

# set empty array
hmap = np.zeros((num_cls, num_cls))

# fill array based on networks predictions
for act in range(num_cls):
    for pred in range(num_cls):
        case_act = hit_miss[hit_miss[:, 1] == act, :]  # = to actual value
        case_pred = case_act[case_act[:, 2] == pred, :]  # = to pred value
        hmap[pred, act] = case_pred.shape[0]

# calculate frequency and round to 2 decimals
hmap_out = (hmap/len(Conf))*100
hmap_out = np.around(hmap_out, decimals=2)

# plot heat map
plt.imshow(hmap_out, cmap=plt.cm.Reds)
plt.ylabel('Predicted Class', fontsize=16)
plt.xlabel('Actual Class', fontsize=16)
plt.title('Class Frequency', fontsize=20)
plt.xticks(np.arange(0, num_cls+1, 1), np.arange(0, num_cls+1, 1))
plt.yticks(np.arange(0, num_cls+1, 1), np.arange(0, num_cls+1, 1))

# plot frequency values inside hmap
for (m, n), label in np.ndenumerate(trout):
    plt.text(n, m, label, fontsize=16, ha='center', va='center', color='black')
cb = plt.colorbar()
cb.set_label(label='Frequency', size=14)
plt.show()
