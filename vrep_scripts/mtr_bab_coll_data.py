import vrep
import sys
import time
import numpy as np
from scipy.misc import imsave
import random
import scipy.io as sio
import scipy
from scipy import ndimage
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
    ret_code, euler_angles = vrep.simxGetObjectOrientation(clientID, base_handle, -1, vrep.simx_opmode_streaming)
    rand = np.random.randint(8)
    act = np.random.randint(5)
    action = (act -2)  * 15
    for iterator in range(num_iterations):
        ret_code, pos = vrep.simxGetObjectPosition(clientID, base_handle, -1, vrep.simx_opmode_oneshot)
        ret_code, velo, angle_velo = vrep.simxGetObjectVelocity(clientID, base_handle, vrep.simx_opmode_oneshot)
        ret_code, euler_angles = vrep.simxGetObjectOrientation(clientID, base_handle, -1, vrep.simx_opmode_buffer)
        if rand == 0:
            rand = np.random.randint(135) + 1
            act = np.random.randint(5)
            action = (act -2)  * 15
            return_val = vrep.simxSetJointTargetVelocity(clientID, left_handle, action, vrep.simx_opmode_oneshot)
            return_val2 = vrep.simxSetJointTargetVelocity(clientID, right_handle, action, vrep.simx_opmode_oneshot_wait)

        collector.append([pos[0], pos[1], pos[2], velo[0], velo[1], velo[2], euler_angles[0],
            euler_angles[1], euler_angles[2], action])
        time.sleep(.005)
        rand += -1

    end(clientID)
    counter = 0
    for element in collector:
        if counter < .15 * int(len(collector)):
            val.append(element)
        elif counter < .3 * int(len(collector)):
            test.append(element)
        else:
            train.append(element)
        counter +=1

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



def main():
    num_iterations = 5000

    for x in range(260,1000):
        train_array, train_label, val_array, val_label, test_array, test_label = get_motor_babbl_data(num_iterations)
        np.save(base_dir + '/data_generated/motor_babble/train/feature/train' + str(x), train_array)
        np.save(base_dir + '/data_generated/motor_babble/train/label/train' + str(x), train_label)

        np.save(base_dir + '/data_generated/motor_babble/val/feature/val' + str(x), val_array)
        np.save(base_dir + '/data_generated/motor_babble/val/label/val' + str(x), val_label)

        np.save(base_dir + '/data_generated/motor_babble/test/feature/test' + str(x), test_array)
        np.save(base_dir + '/data_generated/motor_babble/test/label/test' + str(x), test_label)


if __name__ == '__main__':
    main()
