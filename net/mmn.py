import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .resnet import resnet18 as ResNet
import numpy as np
import sys

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        self.resnet = ResNet(pretrained=True)

        self.resnet.fc = nn.Linear(512, num_class) # reznet18

        self.stgcn = ''
        self.temporal_positions = 15
        self.temporal_rgb_frames = 5

    def forward(self, x_, x_rgb):
    #def forward(self, x_rgb):

        feature = self.stgcn.extract_feature(x_)    #수정
        intensity_s = (feature*feature).sum(dim=1)**0.5

        intensity_s = intensity_s.cpu().detach().numpy()

        feature_s = np.abs(intensity_s)

        feature_s = 255 * (feature_s-feature_s.min()) / (feature_s.max()-feature_s.min())
        N, C, T, V, M = x_.size()

        weight = np.full((N, 1, 225, 45*self.temporal_rgb_frames),0) # full_rgb_crop_sklweight_auto_1
        for n in range(N):
            if feature_s[n, :, :, 0].mean(1).mean(0) > feature_s[n, :, :, 1].mean(1).mean(0):
                for j, v in enumerate([3, 11, 7, 18, 14]):
                    # use TOP 10 values along the temporal dimension
                    feature = feature_s[n, :, v, 0]
                    #print('0 node: '+ str(v)+'\n', feature)
                    temp = np.partition(-feature, self.temporal_positions)
                    #print('feature ', v, ' ', feature, -temp[:15].mean())
                    feature = -temp[:self.temporal_positions].mean()
                    weight[n, 0, 45*j:45*(j+1), :] = feature[np.newaxis, np.newaxis]
            else:
                for j, v in enumerate([3, 11, 7, 18, 14]):
                    # use TOP 10 values along the temporal dimension
                    feature = feature_s[n, :, v, 1]
                    #print('1 node: '+ str(v)+'\n', feature)
                    temp = np.partition(-feature, self.temporal_positions)
                    #print('feature ', v, ' ', feature, -temp[:15].mean())
                    feature = -temp[:self.temporal_positions].mean()
                    weight[n, 0, 45*j:45*(j+1), :] = feature[np.newaxis, np.newaxis]

        weight_cuda = torch.from_numpy(weight).float().cuda()
        weight_cuda = weight_cuda / 127
        #print('weight_cuda',weight_cuda[:,0,0,0].cpu().numpy())

        rgb_weighted = x_rgb * weight_cuda
        #'''

        x = self.resnet.conv1(rgb_weighted)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.resnet.fc(x)

        return out
