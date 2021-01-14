#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:32:11 2020

@author: bruce
"""

import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--protocols', default='xsub', choices={'xsub', 'xset'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

protocol = arg.protocols
alpha = arg.alpha

label = open('./data/ntu120/' + protocol + '/val_label.pkl', 'rb')
label = np.array(pickle.load(label))

r1_2s = open('./results/ntu120/' + protocol + '/joint_result_2sagcn.pkl', 'rb')
r1_2s = list(pickle.load(r1_2s).items())
r2_2s = open('./results/ntu120/' + protocol + '/bone_result_2sagcn.pkl', 'rb')
r2_2s = list(pickle.load(r2_2s).items())

r1_g3d = open('./results/ntu120/' + protocol + '/joint_result_msg3d.pkl', 'rb')
r1_g3d = list(pickle.load(r1_g3d).items())
r2_g3d = open('./results/ntu120/' + protocol + '/bone_result_msg3d.pkl', 'rb')
r2_g3d = list(pickle.load(r2_g3d).items())

r1_gcn = open('./results/ntu120/' + protocol + '/joint_result_stgcn.pkl', 'rb')
r1_gcn = list(pickle.load(r1_gcn).items())
r2_gcn = open('./results/ntu120/' + protocol + '/bone_result_stgcn.pkl', 'rb')
r2_gcn = list(pickle.load(r2_gcn).items())

r3_rgb = open('./results/ntu120/' + protocol + '/rgb_fused_result.pkl', 'rb')
r3_rgb = list(pickle.load(r3_rgb).items())

right_num_11_2s = right_num_11_2s_rgb = right_num_2s = right_num_2s_rgb = 0
right_num_11_g3d = right_num_11_g3d_rgb = right_num_g3d = right_num_g3d_rgb = 0
right_num_11_gcn = right_num_11_gcn_rgb = right_num_gcn = right_num_gcn_rgb = 0
total_num = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]

    _, r33_rgb = r3_rgb[i]
    total_num += 1

    _, r11_2s = r1_2s[i]
    _, r22_2s = r2_2s[i]
    r_2s = r11_2s + r22_2s
    r11_2s_rgb = r11_2s + r33_rgb
    r_2s_rgb = r11_2s + r22_2s + r33_rgb
    r11_2s = np.argmax(r11_2s)
    r11_2s_rgb = np.argmax(r11_2s_rgb)
    r_2s = np.argmax(r_2s)
    r_2s_rgb = np.argmax(r_2s_rgb)
    right_num_11_2s += int(r11_2s == int(l))
    right_num_11_2s_rgb  += int(r11_2s_rgb == int(l))
    right_num_2s += int(r_2s == int(l))
    right_num_2s_rgb += int(r_2s_rgb == int(l))

    _, r11_g3d = r1_g3d[i]
    _, r22_g3d = r2_g3d[i]
    r_g3d = r11_g3d + r22_g3d
    r11_g3d_rgb = r11_g3d + r33_rgb
    r_g3d_rgb = r11_g3d + r22_g3d + r33_rgb
    r11_g3d = np.argmax(r11_g3d)
    r11_g3d_rgb = np.argmax(r11_g3d_rgb)
    r_g3d = np.argmax(r_g3d)
    r_g3d_rgb = np.argmax(r_g3d_rgb)
    right_num_11_g3d += int(r11_g3d == int(l))
    right_num_11_g3d_rgb  += int(r11_g3d_rgb == int(l))
    right_num_g3d += int(r_g3d == int(l))
    right_num_g3d_rgb += int(r_g3d_rgb == int(l))

    _, r11_gcn = r1_gcn[i]
    _, r22_gcn = r2_gcn[i]
    r_gcn = r11_gcn + r22_gcn
    r11_gcn_rgb = r11_gcn + r33_rgb
    r_gcn_rgb = r11_gcn + r22_gcn + r33_rgb
    r11_gcn = np.argmax(r11_gcn)
    r11_gcn_rgb = np.argmax(r11_gcn_rgb)
    r_gcn = np.argmax(r_gcn)
    r_gcn_rgb = np.argmax(r_gcn_rgb)
    right_num_11_gcn += int(r11_gcn == int(l))
    right_num_11_gcn_rgb  += int(r11_gcn_rgb == int(l))
    right_num_gcn += int(r_gcn == int(l))
    right_num_gcn_rgb += int(r_gcn_rgb == int(l))

acc_11_gcn = right_num_11_gcn / total_num
acc_11_gcn_rgb = right_num_11_gcn_rgb / total_num
acc_gcn = right_num_gcn / total_num
acc_gcn_rgb = right_num_gcn_rgb / total_num
print('ST-GCN   Joint: {:0.4f}; Joint+RGB: {:0.4f}; Joint+Bone: {:0.4f}; Joint+Bone+RGB: {:0.4f}'.format(acc_11_gcn,acc_11_gcn_rgb,acc_gcn,acc_gcn_rgb))
acc_11_2s = right_num_11_2s / total_num
acc_11_2s_rgb = right_num_11_2s_rgb / total_num
acc_2s = right_num_2s / total_num
acc_2s_rgb = right_num_2s_rgb / total_num
print('2s-GCN   Joint: {:0.4f}; Joint+RGB: {:0.4f}; Joint+Bone: {:0.4f}; Joint+Bone+RGB: {:0.4f}'.format(acc_11_2s,acc_11_2s_rgb,acc_2s,acc_2s_rgb))
acc_11_g3d = right_num_11_g3d / total_num
acc_11_g3d_rgb = right_num_11_g3d_rgb / total_num
acc_g3d = right_num_g3d / total_num
acc_g3d_rgb = right_num_g3d_rgb / total_num
print('MS-G3D   Joint: {:0.4f}; Joint+RGB: {:0.4f}; Joint+Bone: {:0.4f}; Joint+Bone+RGB: {:0.4f}'.format(acc_11_g3d,acc_11_g3d_rgb,acc_g3d,acc_g3d_rgb))
