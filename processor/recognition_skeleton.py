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
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
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
        # Bruce 20190918
        #self.epoch_info['top {}'.format(k)] = '{:.2f}'.format(100 * accuracy)
        if k==1:
            self.progress_info[int(self.meta_info['epoch']/self.arg.eval_interval), 2]  =  100 * accuracy
            if phase=='eval':   # 모든 eval에서 CM 출력하도록 수정함.
                self.save_recall_precision(self.meta_info['epoch'])
            if accuracy > self.meta_info['best_t1'] and phase=='eval':
                self.meta_info['best_t1'] = accuracy
                self.meta_info['is_best'] = True
                # self.save_recall_precision(self.meta_info['epoch'])
        else:
            self.progress_info[int(self.meta_info['epoch']/self.arg.eval_interval), 3]  =  100 * accuracy

# confusion matrix 생성
    def save_recall_precision(self, epoch): #original input: (label, score),score refers to self.result
        instance_num, class_num = self.result.shape
        rank = self.result.argsort()
        confusion_matrix = np.zeros([class_num, class_num])

        for i in range(instance_num):
            true_l = self.label[i]
            pred_l = rank[i, -1]
            confusion_matrix[true_l][pred_l] += 1
        #np.savetxt("confusion_matrix.csv", confusion_matrix, fmt='%.3e', delimiter=",")
        np.savetxt(os.path.join(self.arg.work_dir,'confusion_matrix_epoch_{}.csv').format(epoch+1), confusion_matrix, fmt='%d', delimiter=",")

        tp = []
        tn = []
        fp = []
        fn = []
        acc = []

        for i in range(class_num):
            true_p = confusion_matrix[i][i]
            false_n = sum(confusion_matrix[i, :]) - true_p
            false_p = sum(confusion_matrix[:, i]) - true_p
            true_n = confusion_matrix.sum() - (false_n + false_p + true_p)
            accuracy = (true_p + true_n) * 1.0 / (true_p + true_n + false_n + false_p)
            tp.append(true_p)
            tn.append(true_n)
            fp.append(false_p)
            fn.append(false_n)
            acc.append(accuracy)

            precision_ = true_p * 1.0 / (true_p + false_p)
            recall_ = true_p * 1.0 / (true_p + false_n)
            # if np.isnan(precision_):
            #     precision_ = 0
            # if np.isnan(recall_):
            #     recall_ = 0
            # precision.append(precision_)
            # recall.append(recall_)
            

        # recall = np.asarray(recall)
        # precision = np.asarray(precision)
        tp = np.asarray(tp)
        tn = np.asarray(tn)
        fp = np.asarray(fp)
        fn = np.asarray(fn)
        acc = np.asarray(acc)

        labels = np.asarray(range(1,class_num+1))
        res = np.column_stack([labels.T, tp.T, tn.T, fp.T, fn.T, acc.T])
        np.savetxt(os.path.join(self.arg.work_dir,'metrics_from_CM_epoch_{}.csv'.format(epoch+1)), res, fmt='%.4e', delimiter=",", header="   Label,      TP,      TN,      FP,      FN,     ACC")


    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        result_frag = []
        label_frag = []
        loss_value = []

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            output = self.model(data)
            ls_cls = self.loss(output, label)

            result_frag.append(output.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())
            # backward
            self.optimizer.zero_grad()
            ls_cls.backward()
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

        for data, label in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())
        print("result_frag", result_frag)
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

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--lambda_l1', type=float, default=0.00001, help='lambda for l1 weight regularization')
        parser.add_argument('--lambda_l2', type=float, default=0.0001, help='lambda for l2 weight regularization')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
