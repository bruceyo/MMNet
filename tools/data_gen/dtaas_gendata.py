## 실행 명령어: python tools/data_gen/dtaas_gendata.py

import sys
sys.path.extend(['../'])

import pickle
import argparse

from tqdm import tqdm

from preprocess import pre_normalization

# 주석이 된 subject는 test subject
training_subjects = [
    1, 2, #3,      # FM
    11, 12, 13,     # FO
    21, 22, #23,     # FY
    31, 32, #33,     # MM
    41, 42, 43,     # MO
    51, 52, 53      # MY
]

training_cameras = [1]

max_body_true = 2
max_body_kinect = 4

num_joint = 25
max_frame = 10000

import numpy as np
import os

def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []

        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
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
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
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
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.txt' for line in f.readlines()]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    file_list = sorted(os.listdir(data_path))
    file_list_txt = [file for file in file_list if file.endswith(".txt")]
    for filename in file_list_txt:
        if filename in ignored_samples:
            continue

        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 3])
        if "FM" in filename:
            subject_id = 0 + int(filename[filename.find('FM') + 3:filename.find('FM') + 5])
        elif "FO" in filename:
            subject_id = 10 + int(filename[filename.find('FO') + 3:filename.find('FO') + 5])
        elif "FY" in filename:
            subject_id = 20 + int(filename[filename.find('FY') + 3:filename.find('FY') + 5])
        elif "MM" in filename:
            subject_id = 30 + int(filename[filename.find('MM') + 3:filename.find('MM') + 5])
        elif "MO" in filename:
            subject_id = 40 + int(filename[filename.find('MO') + 3:filename.find('MO') + 5])
        elif "MY" in filename:
            subject_id = 50 + int(filename[filename.find('MY') + 3:filename.find('MY') + 5])

        camera_id = int("001")

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

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

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    # Fill in the data tensor `fp` one training example a time
    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data

    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='/home/irteam/YJ2/Final_skeletons/')     # 검증할 skeleton(.txt) 파일 경로 => 검증 데이터 경로로 수정해야함.
    parser.add_argument('--ignored_sample_path', default='/home/zio/mocap/NIA-MoCap-1/resource/ignore_files.txt')   # 위의 검증 경로에 있는 파일 중 skeleton이 아닌 파일명 리스트를 작성한 파일
    parser.add_argument('--out_folder', default='/home/zio/mocap/preprocessed_final')  # train/testset이 저장되는 경로

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        if benchmark == 'xview':
            break
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
