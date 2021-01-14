import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

import shutil
# python ensemble.py` --datasets ntu/xprotocol
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='nucla', choices={'nucla'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()
protocols = ['123','132','231']
dataset = arg.datasets
if_save_for_mmnet = False

for protocol in protocols:
    #label = open('../../data/' + dataset + '/' + protocol + '/val_label.pkl', 'rb')
    label = open('./data/' + dataset + '/' + protocol + '/val_label.pkl', 'rb')
    label = np.array(pickle.load(label))

    #r1 = open('../../data/st-gcn/ucla/rgb_only/'+protocol+model+'/best_result.pkl', 'rb')
    r1 = open('./results/nucla/' + protocol + '/rgb_fused_result.pkl', 'rb')
    r1 = list(pickle.load(r1).items())

    r2 = open('./results/nucla/' + protocol + '/joint_result_msg3d.pkl', 'rb')
    r2 = list(pickle.load(r2).items())
    r3 = open('./results/nucla/' + protocol + '/bone_result_msg3d.pkl', 'rb')
    r3 = list(pickle.load(r3).items())

    r2_agcn = open('./results/nucla/' + protocol + '/joint_result_2sagcn.pkl', 'rb')
    r2_agcn = list(pickle.load(r2_agcn).items())
    r3_agcn = open('./results/nucla/' + protocol + '/bone_result_2sagcn.pkl', 'rb')
    r3_agcn = list(pickle.load(r3_agcn).items())

    r2_stgcn = open('./results/nucla/' + protocol + '/joint_result_stgcn.pkl', 'rb')
    r2_stgcn = list(pickle.load(r2_stgcn).items())
    r3_stgcn = open('./results/nucla/' + protocol + '/bone_result_stgcn.pkl', 'rb')
    r3_stgcn = list(pickle.load(r3_stgcn).items())

    right_num_r33 = right_num_r22 = right_num_r11 = right_num_r2_3 = right_num = total_num = right_num_5 = 0
    right_num_r33_agcn = right_num_r22_agcn = right_num_r2_3_agcn = right_num_agcn = 0
    right_num_r33_stgcn = right_num_r22_stgcn = right_num_r2_3_stgcn = right_num_stgcn = 0

    two_skl = []
    for i in tqdm(range(len(label[0]))):
        _, l = label[:, i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        r = r11 + r22 + r33
        r2_3 = r22 + r33
        #r =  r22 + r33
        two_skl.append(r)

        #rank_5 = r.argsort()[-5:]
        #right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        r2_3 = np.argmax(r2_3)
        right_num_r2_3 += int(r2_3 == int(l))
        r11 = np.argmax(r11)
        right_num_r11 += int(r11 == int(l))
        r22 = np.argmax(r22)
        right_num_r22 += int(r22 == int(l))
        r33 = np.argmax(r33)
        right_num_r33 += int(r33 == int(l))

        _, r11 = r1[i]
        _, r22_agcn = r2_agcn[i]
        _, r33_agcn = r3_agcn[i]
        r_agcn = r11 + r22_agcn + r33_agcn
        r2_3_agcn = r22_agcn + r33_agcn
        r_agcn = np.argmax(r_agcn)
        right_num_agcn += int(r_agcn == int(l))
        r2_3_agcn = np.argmax(r2_3_agcn)
        right_num_r2_3_agcn += int(r2_3_agcn == int(l))
        r22_agcn = np.argmax(r22_agcn)
        right_num_r22_agcn += int(r22_agcn == int(l))
        r33_agcn = np.argmax(r33_agcn)
        right_num_r33_agcn += int(r33_agcn == int(l))

        _, r11 = r1[i]
        _, r22_stgcn = r2_stgcn[i]
        _, r33_stgcn = r3_stgcn[i]
        r_stgcn = r11 + r22_stgcn + r33_stgcn
        r2_3_stgcn = r22_stgcn + r33_stgcn
        r_stgcn = np.argmax(r_stgcn)
        right_num_stgcn += int(r_stgcn == int(l))
        r2_3_stgcn = np.argmax(r2_3_stgcn)
        right_num_r2_3_stgcn += int(r2_3_stgcn == int(l))
        r22_stgcn = np.argmax(r22_stgcn)
        right_num_r22_stgcn += int(r22_stgcn == int(l))
        r33_stgcn = np.argmax(r33_stgcn)
        right_num_r33_stgcn += int(r33_stgcn == int(l))

        total_num += 1

    acc = right_num / total_num
    acc_r2_3 = right_num_r2_3 / total_num
    acc_r11 = right_num_r11 / total_num
    acc_r22 = right_num_r22 / total_num
    acc_r33 = right_num_r33 / total_num
    acc_agcn = right_num_agcn / total_num
    acc_r2_3_agcn = right_num_r2_3_agcn / total_num
    acc_r22_agcn = right_num_r22_agcn / total_num
    acc_r33_agcn = right_num_r33_agcn / total_num
    acc_stgcn = right_num_stgcn / total_num
    acc_r2_3_stgcn = right_num_r2_3_stgcn / total_num
    acc_r22_stgcn = right_num_r22_stgcn / total_num
    acc_r33_stgcn = right_num_r33_stgcn / total_num
    #print('protocol ' + protocol )
    print('Protocol ' + protocol + ', RGB: {:0.4f}; joint: {:0.4f}; bone: {:0.4f}; st-gcn: {:0.4f}; MMNet: {:0.4f}.'.format(acc_r11, acc_r22_stgcn, acc_r33_stgcn, acc_r2_3_stgcn, acc_stgcn))
    print('Protocol ' + protocol + ', RGB: {:0.4f}; joint: {:0.4f}; bone: {:0.4f}; 2s-agcn: {:0.4f}; MMNet: {:0.4f}.'.format(acc_r11, acc_r22_agcn, acc_r33_agcn, acc_r2_3_agcn, acc_agcn))
    print('Protocol ' + protocol + ', RGB: {:0.4f}; joint: {:0.4f}; bone: {:0.4f}; ms-g3d: {:0.4f}; MMNet: {:0.4f}.'.format(acc_r11, acc_r22, acc_r33, acc_r2_3, acc))
    print('\n')
