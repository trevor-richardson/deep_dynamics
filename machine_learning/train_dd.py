import time
import numpy as np
import random
import math
import sys
import scipy.io as sio
import os
from os.path import isfile, join
from os import listdir

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import argparse
import configparser
from scipy import stats

config = configparser.ConfigParser()
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']
load_data_bool = False
'''
This supervised learning model learns to predict the future state of the robot given only information retrievable to
A robot in the field such as its internal state. Given the robot state and action predict the next state.
In theory this is a labeling device for an unsupervised scenario which is -- did some external pertubation occur
'''

''' Load Data -- View Statistics -- Preprocess Data '''
''' train_feature, train_label -- 70 percent of hits and misses and their label -- 30 percent I want by ground truth for checking'''

if load_data_bool:
    hit_dir = base_dir + '/data_generated/current_version/hit_state/'
    miss_dir = base_dir + '/data_generated/current_version/miss_state/'

    data = []
    label = []
    ground_truth = []

    hit = [f for f in listdir(hit_dir) if isfile(join(hit_dir, f))]
    miss = [f for f in listdir(miss_dir) if isfile(join(miss_dir, f))]

    for indx, element in enumerate(hit):
        hit[indx] = hit_dir + element
    for indx, element in enumerate(miss):
        miss[indx] = miss_dir + element


    hit_with_label = []
    for element in hit:
        arr = np.load(element)
        adder = []
        for index in range(arr.shape[0] - 1):
            x = np.concatenate((arr[index], arr[index+1, :6]))
            adder.append(x)
        hit_with_label.append(np.asarray(adder))

    miss_with_label = []
    for element in miss:
        arr = np.load(element)
        adder = []
        for index in range(arr.shape[0] - 1):
            x = np.concatenate((arr[index], arr[index+1, :6]))
            adder.append(x)
        miss_with_label.append(np.asarray(adder))

    data = hit_with_label[0]
    for index in range(len(hit_with_label) -1):
        data = np.concatenate((data, hit_with_label[index+1]), axis=0)

    for index in range(len(miss_with_label)):
        data = np.concatenate((data, miss_with_label[index]), axis=0)

    np.random.shuffle(data)

    train_data = data[: int(data.shape[0] * .6), :7]
    train_label = data[: int(data.shape[0] * .6), 7:]


    np.save('train_data', train_data)
    np.save('train_label', train_label)
else:
    train_data = np.load('train_data.npy')
    train_label = np.load('train_label.npy')


#get basic stats about data
axis_0 = train_data[:,0]
print(np.mean(axis_0))
print(np.std(axis_0))
print(axis_0.shape)
new_dat = stats.zscore(axis_0)
new_dat = np.round(new_dat, 2)
print(new_dat[:100])

print("Column 0", np.max(train_data[:,0]), np.min(train_data[:,0]))
print("Column 1", np.max(train_data[:,1]), np.min(train_data[:,1]))
print("Column 2", np.max(train_data[:,2]), np.min(train_data[:,2]))
print("Column 3", np.max(train_data[:,3]), np.min(train_data[:,3]))
print("Column 4", np.max(train_data[:,4]), np.min(train_data[:,4]))
print("Column 5", np.max(train_data[:,5]), np.min(train_data[:,5]))
print("Column 6", np.max(train_data[:,6]), np.min(train_data[:,6]))
axis_0 = train_data[:,1]
print(np.mean(axis_0))
print(np.std(axis_0))
print(axis_0.shape)
new_dat = stats.zscore(axis_0)
new_dat = np.round(new_dat, 2)
print(new_dat[:100])

''' Load Model '''

''' Train Model '''

''' Validate Model '''
