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
from deep_dynamics_model import Deep_Dynamics

config = configparser.ConfigParser()
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']
load_data_bool = False
train_bool = True
model_to_load = '0.158387924063.pth'
sys.path.append(base_dir + '/vrep_scripts/')
from run_vrep_simulation import execute_exp

'''Load DD Model'''

inp_shape = 10
out_shape = 9
model = Deep_Dynamics(inp_shape, 60, 40, 30, 20, 20, out_shape, .3)

if torch.cuda.is_available():
    model.cuda()

def load_model(path):
    global model
    try:
        model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/" + path))
    except ValueError:
        print("Not a valid model to load")
        sys.exit()
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
load_model(model_to_load)

'''Analyze Output From Simulation'''

def evaluate_model(num_forward_passes, single_vid):
    global model
    model.train()
    smallest = 999999999
    rewards = []
    rew = []

    for index in range(int(single_vid.shape[0] -1)):
        lst = []
        input_to_model = torch.from_numpy(single_vid[index])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
        input_to_model = Variable(input_to_model.float(), volatile=True)
        for inner_index in range(num_forward_passes):
            lst.append((model(input_to_model).cpu().data.numpy()))
        rewards.append(calc_statistics(np.asarray(lst), single_vid[index + 1, :9]))
        n1, n2 = calc_indra_2(np.asarray(lst), single_vid[index + 1, :9])
        rew.append(n2)
        del(lst[:])
    return rewards, rew


'''Calculate "probability kinda" of hit'''
def calc_statistics(lst, recorded_state):
    distribution = []
    mean = np.mean(lst, axis=0)
    covar = np.cov(lst, rowvar=False)
    pdf = stats.multivariate_normal.pdf(recorded_state, mean=mean, cov=covar)
    return pdf


def calc_indra_1(lst, recorded_state, mean):
    delta = mean - recorded_state
    return delta, np.linalg.norm(delta)


def calc_indra_2(lst, recorded_state):
    mean = np.mean(lst, axis=0)
    covar = np.cov(lst, rowvar=False)
    delta, norm_delta = calc_indra_1(lst, recorded_state, mean)
    delta_2 = np.exp(delta - 2*covar)
    return norm_delta, np.linalg.norm(delta_2)


''' Main function which calls run_vrep_simulation '''

def main():
    stochastic_forward_passes = 32
    for i in range(20):
        data = execute_exp()

        rewards, rew = evaluate_model(stochastic_forward_passes, data)
        # print(rew)
        print(max(rew))
        print(min(rew))

        # time.sleep(10)


if __name__ == '__main__':
    main()
