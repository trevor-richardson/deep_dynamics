import vrep
import sys
import time
import numpy as np
from scipy.misc import imsave
import random
import scipy.io as sio
import scipy
from scipy import ndimage
import cv2
from time import sleep
import configparser

config = configparser.ConfigParser()
config.read('../config.ini')

base_dir = config['DEFAULT']['BASE_DIR']

'''

This scrip generates the training, validation and testing data for my deep dynamics model

training data
s(t),a(t) -> s(t+1)
save it in a numpy array and then at the end save the numpy array
Have a global variable I iterate each time to represent the current iteration of data collection -- i

'''
def start():
    vrep.simxFinish(-1) # just in case, close all opened connections
    clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) #start my Connection
    #x =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot)
    error_code =vrep.simxStartSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    return clientID, error_code

def end(clientID):
    #end and cleanup
    error_code =vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    return error_code

def get_motor_babbl_data(num_iterations):
    collector = []
    train = []
    val = []
    test = []

    clientID, start_error = start()
    ret_code, left_handle = vrep.simxGetObjectHandle(clientID,'DynamicLeftJoint', vrep.simx_opmode_oneshot_wait)
    ret_code, right_handle = vrep.simxGetObjectHandle(clientID,'DynamicRightJoint', vrep.simx_opmode_oneshot_wait)
    ret_code, base_handle = vrep.simxGetObjectHandle(clientID, 'LineTracerBase', vrep.simx_opmode_oneshot_wait)

    for iterator in range(num_iterations):
        #print state
        ret_code, pos = vrep.simxGetObjectPosition(clientID, base_handle, -1, vrep.simx_opmode_oneshot_wait)
        ret_code, velo, angle_velo = vrep.simxGetObjectVelocity(clientID, base_handle, vrep.simx_opmode_oneshot_wait)

        action = np.random.normal(0, 20.0)

        collector.append([pos[0], pos[1], velo[0], velo[1], action])
        return_val = vrep.simxSetJointTargetVelocity(clientID, left_handle, action, vrep.simx_opmode_oneshot)
        return_val2 = vrep.simxSetJointTargetVelocity(clientID, right_handle, action, vrep.simx_opmode_oneshot_wait)
        sleep(.4) #needs to vary probably
        #print action

    end(clientID)
    for element in collector:
        num = np.random.uniform()
        if num < .15:
            val.append(element)
        elif num < .3:
            test.append(element)
        else:
            train.append(element)

    print("train ", len(train))
    print("val ", len(val))
    print("test ", len(test))

    train_array, train_label, val_array, val_label, test_array, test_label = make_final_format(train, val, test)
    return train_array, train_label, val_array, val_label, test_array, test_label


def make_final_format(train, val, test):
    train_array = []
    train_label = []
    val_array  = []
    val_label = []
    test_array = []
    test_label = []

    for index in range(len(train) - 1):
        train_array.append(train[index])
        train_label.append(train[index + 1][0:len(train[index])-1])
        # print (train_array[index], " PRINT THIS ", train_label[index])

    for index in range(len(val) - 1):
        val_array.append(val[index])
        val_label.append(val[index + 1][0:len(val[index])-1])

    for index in range(len(test) - 1):
        test_array.append(test[index])
        test_label.append(test[index + 1][0:len(test[index])-1])


    train_array = np.asarray(train_array)
    train_label = np.asarray(train_label)
    val_array = np.asarray(val_array)
    val_label = np.asarray(val_label)
    test_array = np.asarray(test_array)
    test_label = np.asarray(test_label)
    return train_array, train_label, val_array, val_label, test_array, test_label

'''
prevsim represents the number of times I've ran this to make more data
iter_start and end are what decides how many data points I will create this iteration
'''

def main(prevsim):
    num_iterations = 2000
    version = 0 #increment this every time so you do not overwrite the data
    train_array, train_label, val_array, val_label, test_array, test_label = get_motor_babbl_data(num_iterations)
    np.save(base_dir + '/data_generated/motor_babble/train/feature/train' + str(version), train_array)
    np.save(base_dir + '/data_generated/motor_babble/train/label/train' + str(version), train_label)

    np.save(base_dir + '/data_generated/motor_babble/val/feature/val' + str(version), val_array)
    np.save(base_dir + '/data_generated/motor_babble/val/label/val' + str(version), val_label)

    np.save(base_dir + '/data_generated/motor_babble/test/feature/test' + str(version), test_array)
    np.save(base_dir + '/data_generated/motor_babble/test/label/test' + str(version), test_label)


if __name__ == '__main__':
    prevsim = 0
    main(prevsim)
