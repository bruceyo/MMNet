#!/usr/bin/env python
# pylint: disable=W0201
import os
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

import sys
import matplotlib.pyplot as plt
from PIL import Image
import time
import pickle

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        print('Load model st-gcn')
        self.model.stgcn = self.io.load_model('net.gcn.Model',
                                              **(self.arg.model_args))
        self.model.temporal_positions = self.arg.temporal_positions
        self.model.temporal_rgb_frames = self.arg.test_feeder_args['temporal_rgb_frames']

        self.model.stgcn = self.io.load_weights(self.model.stgcn, self.arg.joint_weights,
                                          self.arg.ignore_weights)
        if self.arg.fix_weights:
            self.model.stgcn.eval()
            print('Load model st-gcn  DONE')

        #self.model.apply(weights_init)

        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                #Over here we want to only update the parameters of the classifier so
                #self.model.module.classifier.parameters(),
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k, phase):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))
        acc_r1r3 = acc_stgcn = acc = acc5 = 0
        #'''
        # *********************ensemble with skl results***********************start
        if phase == 'eval':
            r1_yan = open(self.arg.skeleton_joints_pkl, 'rb')
            r1_yan = list(pickle.load(r1_yan).items())
            r2_yan = open(self.arg.skeleton_bones_pkl, 'rb')
            r2_yan = list(pickle.load(r2_yan).items())

            result_dict = dict(
                zip(self.data_loader['test'].dataset.sample_name,
                    self.result))
            self.io.save_pkl(result_dict, 'tmp_test_result.pkl')
            r3 = open(os.path.join(self.arg.work_dir,'tmp_test_result.pkl'), 'rb')
            r3 = list(pickle.load(r3).items())
            #r2 = self.result.items()
            right_num = total_num = right_num_5 = right_num_r1r3 = right_num_stgcn = 0
            for i,l in enumerate(self.label):
                #_, l = label[i]
                #_, r11 = r1[i]
                #_, r22 = r2[i]
                _, r33 = r3[i]
                _, r11_yan = r1_yan[i]
                _, r22_yan = r2_yan[i]
                #r = r11 + r22 + r33

                r1r3 = r11_yan + r33
                r_stgcn = r11_yan + r22_yan + r33
                rank_5 = r1r3.argsort()[-5:]
                right_num_5 += int(int(l) in rank_5)
                #r = np.argmax(r)
                r1r3 = np.argmax(r1r3)
                r_stgcn = np.argmax(r_stgcn)

                #right_num += int(r == int(l))
                right_num_r1r3 += int(r1r3 == int(l))
                right_num_stgcn += int(r_stgcn == int(l))
                total_num += 1
            #acc = right_num / total_num
            acc_r1r3 = right_num_r1r3 / total_num
            acc_stgcn = right_num_stgcn / total_num
            acc5 = right_num_5 / total_num
            #accuracy = acc
            self.io.print_log('ST-ROI Top 1:                        {}'.format(accuracy))
            self.io.print_log('Top 1 with Joint:                    {}'.format(acc_r1r3))
            self.io.print_log('Top 1 with Joint + Bone (ST-GCN):    {}'.format(acc_stgcn))
        # *********************ensemble with skl results***********************end

        if k==1:
            self.progress_info[int(self.meta_info['epoch']/self.arg.eval_interval), 2]  =  100 * accuracy
            #if (accuracy > self.meta_info['best_t1'] or acc_stgcn > self.meta_info['best_t1']) and phase=='eval':
            if acc_stgcn > self.meta_info['best_t1'] and phase=='eval':
                self.meta_info['best_t1'] = acc_stgcn
                self.meta_info['is_best'] = True
                self.io.print_log('Best Ensemble Top 1: {}; Top 1 with STGCN: {}'.format(acc_r1r3, acc_stgcn))
        else:
            self.progress_info[int(self.meta_info['epoch']/self.arg.eval_interval), 3]  =  100 * acc_r1r3

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        result_frag = []
        label_frag = []
        loss_value = []

        for data, rgb, label in loader:

            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            rgb = rgb.float().to(self.dev)

            output = self.model(data,rgb)

            ls_cls = self.loss(output, label)
            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())
            loss = ls_cls

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['ls_cls'] = ls_cls.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['ls_cls'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['ls_cls']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        if ((self.meta_info['epoch'] + 1) % self.arg.eval_interval == 0):
            for k in self.arg.show_topk:
                self.show_topk(k, 'train')


    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, rgb, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            rgb = rgb.float().to(self.dev)

            with torch.no_grad():
                output = self.model(data, rgb)

            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                ls_cls = self.loss(output, label)
                loss = ls_cls
                self.iter_info['ls_cls'] = ls_cls.data.item()
                loss_value.append(ls_cls.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)

        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['ls_cls']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k, 'eval')


    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--fix_weights', type=str2bool, default=True, help='set the teacher in evaluation mode')
        parser.add_argument('--joint_weights', default=None, help='the learned weights of the teacher network')
        parser.add_argument('--temporal_positions', default=None, help='temporal positions for calculating the joint weights')

        return parser
