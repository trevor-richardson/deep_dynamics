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

config = configparser.ConfigParser()
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']

'''
This supervised learning model learns to predict the future state of the robot given only information retrievable to
A robot in the field such as its internal state. Given the robot state and action predict the next state.
In theory this is a labeling device for an unsupervised scenario which is -- did some external pertubation occur
'''

''' Load Data -- View Statistics -- Preprocess Data '''
''' train_feature, train_label -- 70 percent of hits and misses and their label -- 30 percent I want by ground truth for checking'''
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

print("Num Hit Videos", len(hit))
print("Num Miss Videos", len(miss))

#need to load each numpy -- append next state features onto current feature -- delete last feature



#Load all of the data into numpy arrays cut the first seventy



''' Load Model '''



''' Train Model '''



''' Validate Model '''
