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
model_to_load = '0.111379066878.pth'

'''
This supervised learning model learns to predict the future state of the robot given only information retrievable to
A robot in the field such as its internal state. Given the robot state and action predict the next state.
In theory this is a labeling device for an unsupervised scenario which is -- did some external pertubation occur
'''

''' Load Data -- View Statistics -- Preprocess Data '''
''' train_feature, train_label -- 70 percent of hits and misses and their label -- 30 percent I want by ground truth for checking'''

if load_data_bool:
    hit_dir = base_dir + '/data_generated/aligned_version/hit_state/'
    miss_dir = base_dir + '/data_generated/aligned_version/miss_state/'

    ground_truth = []

    hit = [f for f in listdir(hit_dir) if isfile(join(hit_dir, f))]
    miss = [f for f in listdir(miss_dir) if isfile(join(miss_dir, f))]

    for indx, element in enumerate(hit):
        hit[indx] = hit_dir + element
    for indx, element in enumerate(miss):
        miss[indx] = miss_dir + element

    #here is where the split needs to occur
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
    for index in range(int(len(hit_with_label) *.7)):
        data = np.concatenate((data, hit_with_label[index+1]), axis=0)

    for index in range(int(len(miss_with_label) * .7)):
        data = np.concatenate((data, miss_with_label[index]), axis=0)

    #save the last 30 percent of data
    hitss = np.asarray(hit_with_label[:int(len(hit_with_label) *.7)])
    misses = np.asarray(miss_with_label[:int(len(miss_with_label) * .7)])
    np.save('./preprocess_data/test_hit', hitss)
    np.save('./preprocess_data/test_miss', misses)
    print(hitss.shape)
    print(hitss[0].shape)
    print(misses.shape)

    np.random.shuffle(data)

    train_data = data[:, :7]
    train_label = data[:, 7:]
    print(train_data.shape)
    print(train_label.shape, "yup train")
    print(len(hit_with_label))

    np.save('./preprocess_data/train_data', train_data)
    np.save('./preprocess_data/train_label', train_label)
else:
    data = np.load(base_dir + '/data_generated/motor_babble/train/feature/train0.npy')
    label = np.load(base_dir + '/data_generated/motor_babble/train/label/train0.npy')
    val_data = np.load(base_dir + '/data_generated/motor_babble/val/feature/val0.npy')
    val_label = np.load(base_dir + '/data_generated/motor_babble/val/label/val0.npy')
    test_data = np.load(base_dir + '/data_generated/motor_babble/test/feature/test0.npy')
    test_label = np.load(base_dir + '/data_generated/motor_babble/test/label/test0.npy')

    for i in range(100):
        data2 = np.load(base_dir + '/data_generated/motor_babble/train/feature/train' + str(i) + '.npy')
        label2 = np.load(base_dir + '/data_generated/motor_babble/train/label/train' + str(i) + '.npy')
        val_data2 = np.load(base_dir + '/data_generated/motor_babble/val/feature/val' + str(i) + '.npy')
        val_label2 = np.load(base_dir + '/data_generated/motor_babble/val/label/val' + str(i) + '.npy')
        test_data2 = np.load(base_dir + '/data_generated/motor_babble/test/feature/test' + str(i) + '.npy')
        test_label2 = np.load(base_dir + '/data_generated/motor_babble/test/label/test' + str(i) + '.npy')
        data = np.concatenate((data, data2), axis=0)
        label = np.concatenate((label, label2), axis=0)
        val_data = np.concatenate((val_data, val_data2), axis=0)
        val_label = np.concatenate((val_label, val_label2), axis=0)
        test_data = np.concatenate((test_data, test_data2), axis=0)
        test_label = np.concatenate((test_label, test_label2), axis=0)

    data = np.round(data, 3)

    label = np.round(label, 3)
    print(data.shape, label.shape, "Training sizes")
    label = np.round(label, 3)
    test_data = np.round(test_data, 3)
    test_label = np.round(test_label, 3)
    val_data = np.round(val_data, 3)
    val_label = np.round(val_label, 3)


''' Load Model '''
model = Deep_Dynamics(int(data.shape[1]), 60, 40, 30, 20, 20, int(label.shape[1]), .3)
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=.0001)

''' Train Model '''
def train_model(epoch, batch_size):
    model.train()
    global data
    global label
    train_loss = 0
    step_counter = 0

    for iteration in range(int(int(data.shape[0])/batch_size)):
        input_to_model = torch.from_numpy(data[(iteration * batch_size):((iteration+1)*batch_size)])
        y_ = torch.from_numpy(label[(iteration * batch_size):((iteration+1)*batch_size)])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
            y_ = y_.cuda()
        input_to_model = Variable(input_to_model.float())
        y_ = Variable(y_.float())

        pred = model(input_to_model)

        loss = F.mse_loss(pred, y_)

        loss.backward()
        optimizer.step()
        train_loss+=loss.data
        step_counter +=1

    print('Train Epoch: {}\tLoss: {:.6f}'.format(
        epoch, train_loss.cpu().numpy()[0]/step_counter))

''' Validate Model '''
def validate_model(epoch, batch_size):
    model.eval()
    global val_data
    global val_label
    test_loss = 0
    step_counter = 0

    for iteration in range(int(int(val_data.shape[0])/batch_size)):
        input_to_model = torch.from_numpy(val_data[(iteration * batch_size):((iteration+1)*batch_size)])
        y_ = torch.from_numpy(val_label[(iteration * batch_size):((iteration+1)*batch_size)])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
            y_ = y_.cuda()
        input_to_model = Variable(input_to_model.float(), volatile=True)
        y_ = Variable(y_.float())

        pred = model(input_to_model)

        loss = F.mse_loss(pred, y_)

        test_loss+=loss.data
        step_counter +=1

    print('Test Epoch: {}\tLoss: {:.6f}'.format(
        epoch, test_loss.cpu().numpy()[0]/step_counter))

    return test_loss.cpu().numpy()[0]/step_counter


''' Helper functions '''
def save_model(model, loss):
    torch.save(model.state_dict(), base_dir + '/machine_learning/saved_models/' + str(loss) + '.pth')


def load_model(path):
    global model
    try:
        model.load_state_dict(torch.load(base_dir + "/machine_learning/saved_models/" + path))
    except ValueError:
        print("Not a valid model to load")
        sys.exit()

#load 70% of the data for training save best validation error
'''Test what strategy allows me to classify hits vs misses'''
def evaluate_model(num_forward_passes, single_hit, single_miss):
    print("evaluating strategy")
    model.train()

    smallest = 999999999

    for index in range(int(single_hit.shape[0])):
        lst = []
        input_to_model = torch.from_numpy(single_hit[index, :7])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
        input_to_model = Variable(input_to_model.float(), volatile=True)
        for inner_index in range(num_forward_passes):
            lst.append((model(input_to_model).cpu().data.numpy()))
        small = calc_statistics(np.asarray(lst), single_hit[index, 7:])
        if smallest > small:
            smallest = small
        del(lst[:])
    print(smallest, "hit")

    smallest = 9999
    for index in range(int(single_miss.shape[0])):
        lst = []
        input_to_model = torch.from_numpy(single_miss[index, :7])
        if torch.cuda.is_available():
            input_to_model = input_to_model.cuda()
        input_to_model = Variable(input_to_model.float(), volatile=True)
        for inner_index in range(num_forward_passes):
            lst.append((model(input_to_model).cpu().data.numpy()))
        small = calc_statistics(np.asarray(lst), single_miss[index, 7:])
        if smallest > small:
            smallest = small
        del(lst[:])
    print(smallest, "miss")


'''Calculate "probability kinda" of hit'''
def calc_statistics(lst, recorded_state):
    distribution = []
    mean = np.mean(lst)
    var = np.var(lst)
    f = stats.multivariate_normal.pdf(recorded_state, mean=mean, cov=var)
    return np.min(f)


def main():
    global model
    global test_hit
    global test_miss
    global model_to_load

    epochs = 1000
    batch_size = 128

    smallest_loss = 10000
    num_forward_passes = 64
    if train_bool:
        for epoch in range(epochs):
            train_model(epoch, batch_size)
            loss = validate_model(epoch, batch_size)
            if loss < 5.0:
                save_model(model, loss)
                smallest_loss = loss
            print("\n*****************************\n")
        print(smallest_loss)
        save_model(model, loss)
    else:
        print("Testing ")
        load_model(model_to_load)
        for i in range(100):
            single_hit = np.round(test_hit[i], 2)
            single_miss = np.round(test_miss[i], 2)

            evaluate_model(num_forward_passes, single_hit, single_miss)

if __name__ == '__main__':
    main()
