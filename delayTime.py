# -*- coding: UTF-8 -*-
import pickle
import cPickle
import numpy as np
import sys
import os
import math

def generate_train_for_new_features(index, fold, maneuver_type, interval_time):
    # input maneuver_type:{lane, turns, all, all_new_features}
    # input interval_time:must be n times 0.04 sec, n belongs to [1,25]
    f1 = open('./checkpoints/' +
              maneuver_type + '/' + fold + '/train_data_' + index + '.pik', 'rb')
    data1 = pickle.load(f1)
    label1 = data1['labels']
    feature1 = data1['features']
    params1 = data1['params']
    delta_frame = 20
    #插值扩展
    label2 = np.zeros(((label1.shape[0]-1)*delta_frame+1, label1.shape[1]))
    feature2 = np.zeros(((label1.shape[0]-1)*delta_frame+1, label1.shape[1], feature1.shape[2]))
    for i in range(0, label2.shape[0]):
        #delta_frame=20,即每隔20帧（0.8s）进行一次采样
        ii = float(i)/20
        label2[i, :] = label1[int(round(i/20)), :]
        feature2[i, :, :] = feature1[int(math.floor(ii)), :, :] - (feature1[int(math.floor(ii)), :, :] - feature1[int(math.ceil(ii)), :, :])*(ii-math.floor(ii))
    begin=label2.shape[0]-1
    #对扩展后的label和feature进行采样
    while begin>=interval_time:
        begin-=interval_time
    index_list=[]
    while begin<label2.shape[0]:
        index_list.append(begin)
        begin+=interval_time
    label3=label2[index_list, :]
    feature3=feature2[index_list, :, :]

    label3 = label3.astype(label1.dtype)
    feature3 = feature3.astype(feature1.dtype)
    params2 = params1
    params2['min_length_sequence'] = 4
    params2['extra_examples'] = 4
    f1.close()
    results2 = {}
    results2['labels'] = label3
    results2['features'] = feature3
    results2['params'] = params2
    results2['actions'] = data1['actions']
    cPickle.dump(results2, open('./checkpoints/'+ maneuver_type + '/' +
                                fold + '/train_data_' + index + str(interval_time) + '.pik', 'wb'))



def generate_test_for_new_features(index, fold, maneuver_type, interval_time):
    # input type:{lane_new_features, turns_new_features}
    f1 = open('./checkpoints/' +
              maneuver_type + '/' + fold + '/test_data_' + index + '.pik', 'rb')
    data1 = pickle.load(f1)
    label1 = data1['labels']
    feature1 = data1['features']
    label1 = np.array(label1)
    feature1 = np.array(feature1)
    delta_frame = 20
    label2 = np.zeros((label1.shape[0], (label1.shape[1]-1)*delta_frame+1, label1.shape[2]))
    feature2 = np.zeros((feature1.shape[0], (feature1.shape[1]-1)*delta_frame+1, feature1.shape[2], feature1.shape[3]))
    for i in range(0, label2.shape[1]):
        ii = float(i)/20
        label2[:, i, :] = label1[:, int(round(i/20)), :]
        feature2[:, i, :, :] = feature1[:, int(math.floor(ii)), :, :] - (feature1[:, int(math.floor(ii)), :, :] - feature1[:, int(math.ceil(ii)), :, :])*(ii-math.floor(ii))
    begin = label2.shape[1] - 1
    while begin >= interval_time:
        begin -= interval_time
    index_list = []
    while begin < label2.shape[1]:
        index_list.append(begin)
        begin += interval_time
    label3 = label2[:, index_list, :]
    feature3 = feature2[:, index_list, :, :]
    label3 = label3.astype(label1.dtype)
    feature3 = feature3.astype(feature1.dtype)
    f1.close()
    results3 = {}
    results3['labels'] = label3
    results3['features'] = feature3
    results3['actions'] = data1['actions']
    cPickle.dump(results3, open('./checkpoints/' + maneuver_type + '/' +
                                fold + '/test_data_' + index + str(interval_time) +'.pik', 'wb'))


def generate_delay_train_time(index, fold, type, interval_time):
    # input type:{all,lane,turn,all_new_features}
    f1 = open('./checkpoints/' +
              type + '/' + fold + '/train_data_' + index + str(interval_time) +'.pik', 'rb')
    data1 = pickle.load(f1)
    label1 = data1['labels']
    feature1 = data1['features']
    params1 = data1['params']
    params2 = params1
    params2['min_length_sequence'] = 4
    params2['extra_examples'] = 4
    # first model{min_len:5,extra_example:5}
    # second_model{min_len:5,extra_example:4}
    #params2 = params2.astype(params1.dtype)
    print(label1.shape)
    print(feature1.shape)
    
    label2 = label1[1:, :]
    label2 = label2.astype(label1.dtype)
    feature2 = np.zeros(
        (feature1.shape[0] - 1, feature1.shape[1], feature1.shape[2]))
    feature2[:, :, 0:-4] = feature1[1:, :, 0:-4]
    feature2[:, :, -4:] = feature1[0:-1, :, -4:]
    feature2 = feature2.astype(feature1.dtype)
    f1.close()
    results = {}                         
    results['labels'] = label2
    results['features'] = feature2
    results['params'] = params2
    results['actions'] = data1['actions']
    cPickle.dump(results, open('./checkpoints/' +
                               type + '_zhouxq/' + fold + '/train_data_' + index + str(interval_time) + '.pik', 'wb'))


def generate_delay_test_time(index, fold, type, interval_time):

    f1 = open('./checkpoints/' +
              type + '/' + fold + '/test_data_' + index + str(interval_time) +'.pik', 'rb')
    data1 = pickle.load(f1)
    label1 = data1['labels']
    feature1 = data1['features']
    label1 = np.array(label1)
    feature1 = np.array(feature1)
    print(label1.shape)
    print(feature1.shape)
    label2 = label1[:, 1:, :]
    label2 = label2.astype(label1.dtype)
    feature2 = np.zeros(
        (feature1.shape[0], feature1.shape[1] - 1, feature1.shape[2], feature1.shape[3]))
    feature2[:, :, :, 0:-4] = feature1[:, 1:, :, 0:-4]
    feature2[:, :, :, -4:] = feature1[:, 0:-1, :, -4:]
    feature2 = feature2.astype(feature1.dtype)
    f1.close()
    results = {}
    results['labels'] = label2
    results['features'] = feature2
    results['actions'] = data1['actions']
    cPickle.dump(results, open('./checkpoints/' +
                               type + '_zhouxq/' + fold + '/test_data_' + index + str(interval_time) + '.pik', 'wb'))


if __name__ == "__main__":

    folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
    maneuver_type = sys.argv[1]
    if maneuver_type == 'all':
        index = ['356988', '356988', '356988', '356988', '356988']
    elif maneuver_type == 'all_new_features':
        index = ['846483', '846483', '846483', '846483', '846483']
    elif maneuver_type == 'lane':
        index = ['723759', '723759', '723759', '723759', '723759']
    elif maneuver_type == 'turns':
        index = ['209221', '209221', '209221', '209221', '209221']
    # the following is to generate train and test file for lane and turn new features
    interval_time = int(sys.argv[2])
    for i, f in zip(index, folds):
        print(i, f)
        generate_train_for_new_features(i, f, maneuver_type, interval_time)
        generate_test_for_new_features(i, f, maneuver_type, interval_time)
    for i, f in zip(index, folds):
        print(i, f)
        generate_delay_train_time(i, f, maneuver_type, interval_time)
        generate_delay_test_time(i, f, maneuver_type, interval_time)

