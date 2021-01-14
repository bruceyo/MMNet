import os
import sys
import pickle

import argparse
import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap

from data_gen.preprocess import pre_normalization

view_dic = {1:"L", 2: "M", 3:"R"}

max_body_true = 2
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30

def read_skeleton(file):
    #print(file)
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(float(f.readline()))
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                if m==0:
                    body_info_key = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_info = {
                        k: float(v)
                        for k, v in zip(body_info_key, f.readline().split())
                    }

                    body_info['numJoint'] = int(f.readline())
                else:
                    body_info_key = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_info = {
                        k: float(v)
                        for k, v in zip(body_info_key, '6 1 0 0 1 1 0 -0.437266 -0.117168 2'.split())
                    }
                    body_info['numJoint'] = 25
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z'#, 'depthX', 'depthY', 'colorX', 'colorY',
                        #'orientationW', 'orientationX', 'orientationY',
                        #'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence

def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s

def read_xyz(file, max_body=4, num_joint=25):
    seq_info = read_skeleton(file)
    if seq_info['numFrame'] > 300:
        start = (seq_info['numFrame'] - 300) // 2
        end = 300 + (seq_info['numFrame'] - 300) // 2
        seq_info['frameInfo'] = [f for n, f in enumerate(seq_info['frameInfo']) if n in range(start,end)]
        seq_info['numFrame'] = 300
    #data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    #data[:, n, j, m] = [v['x'], v['y'], v['z']]
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass
    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    # To (C,T,V,M)
    data = data.transpose(3, 1, 2, 0)
    return data

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    training_split_file_folder = '/mnt/nas/ntu-rgbd/PKUMMD/Split/'
    if benchmark=='xview':
        training_split_file = training_split_file_folder + "cross-view.txt"
    else:
        training_split_file = training_split_file_folder + "cross-subject.txt"
    f = open(training_split_file, "r")
    f.readline()
    training_split = f.readline()
    f.readline()
    testing_split = f.readline()

    training_set = training_split.split(',')
    training_set = [s.strip() for s in training_set]
    training_set = [s for s in training_set if s]

    testing_set = testing_split.split(',')
    testing_set = [s.strip() for s in testing_set]
    testing_set = [s for s in testing_set if s]

    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        file_id = int(
            filename[filename.find('F') + 1:filename.find('F') + 4])
        view_id = int(
            filename[filename.find('V') + 1:filename.find('V') + 4])
        action_class = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        split_train_test_name = '0'+filename[filename.find('F') + 1:filename.find('F') + 4] + '-' + view_dic[view_id]

        if (split_train_test_name in training_set) and (split_train_test_name in testing_set):
            raise ValueError()
        istraining = (split_train_test_name in training_set)

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)
    '''
    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()
    '''
    # Create data tensor (N,C,T,V,M)
    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    # Fill in the data tensor `fp` one training example a time
    for i, s in enumerate(tqdm(sample_name, dynamic_ncols=True)):
        #data = read_xyz(s, max_body=max_body, num_joint=num_joint)
        data = read_xyz(
            os.path.join(data_path, s), max_body=4, num_joint=num_joint)
        # Fill (C,T,V,M) to data tensor (N,C,T,V,M)
        fp[i, :, 0:data.shape[1], :, :] = data

    # Perform preprocessing on data tensor
    fp = pre_normalization(fp)
    # Save input data (train/val)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='/media/bruce/2Tssd/data/pku_skeleton_ntu')
    parser.add_argument(
        '--ignored_sample_path',
        default='/media/bruce/2T/projects/st-gcn_rgb/resource/PKUMMD/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/media/bruce/2Tssd/data/pku_ms_g3d')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
