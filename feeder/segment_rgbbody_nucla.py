#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:23:37 2019

@author: bruce
"""
import os, fnmatch
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

frame_path = '/media/bruce/2Tssd2/NWestern_UCLA/multiview_action/'
openpose_path = '../../data/openpose_ucla/'
save_path = '/media/bruce/2Tssd2/NWestern_UCLA/ucla_rgb_frames_crop/fivefs_240/'
debug = False
duplicate_list=[]

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def rgb_frames(file_folder, sample):
    #sample = 'a01_s01_e00'
    skl_list = file_folder + sample + '/fileList.txt'
    #skl_list = np.loadtxt(skl_list, comments=None)
    skl_list_ = []
    with open(skl_list, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            line = line.split()
            if len(line) > 3:
                line = line[2:5]
            if len(line) < 3:  # filter out the unconsistent frames
                continue
            skl_list_.append(line)

    skl_list_denoise = []
    duplicate = 0
    for i in range(0, len(skl_list_)-1):
        if i == duplicate:
            continue
        if skl_list_[i][0] == skl_list_[i+1][0]:
            skl_list_denoise.append(skl_list_[i+1])
            duplicate = i+1
            if sample not in duplicate_list:
                #print('Duplicate: ', sample)
                duplicate_list.append(sample)
        else:
            skl_list_denoise.append(skl_list_[i])
            duplicate = i

    skl_list = np.array(skl_list_denoise)

    if skl_list.shape[0] < 1:
        empty_list.append(sample)
        print('empty: ', sample)
        return ''

    images = []
    image_folder = file_folder + '/'+ sample

    for skl in range(0, skl_list.shape[0]):
         # frame_167_tc_90467351_skeletons.txt
         skl_file = 'frame_'+str(int(skl_list[skl][0]))+'_tc_'+str(int(skl_list[skl][1]))+'_rgb.jpg'
         images.append(skl_file)
    return images

def filename_construct(subject_id, action_id, environment_id, view_id):
    skeleton_file_name = ''
    rgb_folder_name = ''

    if subject_id/10 >= 1:
        skeleton_file_name = skeleton_file_name +'S' + str(subject_id)
    else:
        skeleton_file_name = skeleton_file_name + 'S0' +  str(subject_id)

    if action_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'A' +  str(action_id)
    else:
        skeleton_file_name = skeleton_file_name + 'A0' +  str(action_id)
    if environment_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'E' +  str(environment_id)
    else:
        skeleton_file_name = skeleton_file_name + 'E0' +  str(environment_id)
    if view_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'V' +  str(view_id)
    else:
        skeleton_file_name = skeleton_file_name + 'V0' +  str(view_id)

    if action_id > 8:
        action_id = action_id + 2
    if action_id > 6 and action_id < 9:
        action_id = action_id + 1
    rgb_folder_name = rgb_folder_name + 'view_' +  str(view_id)
    if action_id/10 >= 1:
        rgb_folder_name = rgb_folder_name + '/a' +  str(action_id)
    else:
        rgb_folder_name = rgb_folder_name + '/a0' +  str(action_id)
    if subject_id/10 >= 1:
        rgb_folder_name = rgb_folder_name +'_s' + str(subject_id)
    else:
        rgb_folder_name = rgb_folder_name + '_s0' +  str(subject_id)
    environment_id = environment_id - 1
    if environment_id/10 >= 1:
        rgb_folder_name = rgb_folder_name + '_e' +  str(environment_id)
    else:
        rgb_folder_name = rgb_folder_name + '_e0' +  str(environment_id)

    return skeleton_file_name, rgb_folder_name

def openposeFile(frame_file, frame, skeleton_file_name, openpose_path):
    frame_file_ = frame_file + '/' + str(frame) + '.jpg'

    frame_ = '';
    if frame/100 >= 1:
        frame_ = str(frame)
    elif frame/10 >= 1:
        frame_ = '0' + str(frame)
    else:
        frame_ ='00' + str(frame)
    openpose_file_ = openpose_path + skeleton_file_name + '/' + skeleton_file_name + '_000000000'+ frame_ + '_keypoints.json'

    return openpose_file_#, frame_file_

def cropBody(openpose_file, frame_file, action_id, flip):
    #upper=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #lower=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #whole=Image.new( 'RGB' , (224,448) , (0,0,0) )

    frame = Image.open(frame_file)

    if openpose_file:
        with open(openpose_file, 'r') as f:
            skeleton = json.load(f)

    # calculate which people?
    if len(skeleton['people']) == 1: # or action_id > 49:
        if len(skeleton['people']) < 1:
            return ''
        head_x = skeleton['people'][0]['pose_keypoints_2d'][0]
        head_y = skeleton['people'][0]['pose_keypoints_2d'][1]
        L_hand_x = skeleton['people'][0]['pose_keypoints_2d'][12]
        L_hand_y = skeleton['people'][0]['pose_keypoints_2d'][13]
        R_hand_x = skeleton['people'][0]['pose_keypoints_2d'][21]
        R_hand_y = skeleton['people'][0]['pose_keypoints_2d'][22]
        L_leg_x = skeleton['people'][0]['pose_keypoints_2d'][30]
        L_leg_y = skeleton['people'][0]['pose_keypoints_2d'][31]
        R_leg_x = skeleton['people'][0]['pose_keypoints_2d'][39]
        R_leg_y = skeleton['people'][0]['pose_keypoints_2d'][40]

        head = frame.crop((head_x-24, head_y - 24, head_x + 24, head_y + 24)) #2*3+1
        L_hand = frame.crop((L_hand_x-24, L_hand_y - 24, L_hand_x + 24, L_hand_y + 24))
        R_hand = frame.crop((R_hand_x-24, R_hand_y - 24, R_hand_x + 24, R_hand_y + 24))
        L_leg = frame.crop((L_leg_x-24, L_leg_y - 24, L_leg_x + 24, L_leg_y + 24))
        R_leg = frame.crop((R_leg_x-24, R_leg_y - 24, R_leg_x + 24, R_leg_y + 24))

        frame_concat=Image.new( 'RGB' , (48,240) , (0,0,0) )
        #print('file consistent: ',openpose_file_)
        if flip:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(R_hand, (0,48))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(R_leg, (0,144))
            frame_concat.paste(L_leg, (0,192))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,48))
            frame_concat.paste(R_hand, (0,96))
            frame_concat.paste(L_leg, (0,144))
            frame_concat.paste(R_leg, (0,192))

        #print('frame_concat   if')
        return frame_concat

    elif len(skeleton['people']) > 1:
        #print(frame_file)
        query_file = frame_file.split('/')
        query_folder = frame_file[0:len(frame_file)-len(query_file[len(query_file)-1])]
        query_file = query_file[len(query_file)-1]
        query_file = query_file.split('_')
        query_file = query_file[0] + '_' + query_file[1] + '_' + query_file[2] + '_' + "*" + '_depth.png'

        depth_file = find(query_file, query_folder)
        if len(depth_file) < 1:
            print(frame_file)
            print(query_file, query_folder)
        depth_frame = Image.open(depth_file[0])
        depth_frame_arr = np.fromiter(iter(depth_frame.getdata()), np.uint16)
        depth_frame_arr.resize(depth_frame.height, depth_frame.width)

        people_dist_min = 4000
        people_index = 0
        joint = 8
        for p in range(len(skeleton['people'])):
            #for i in []: # openpose joint 2, 5, 8, 11
            x = int(skeleton['people'][p]['pose_keypoints_2d'][(joint+1)*3-3]//2)
            y = int(skeleton['people'][p]['pose_keypoints_2d'][(joint+1)*3-2]//2)
            k = 0
            if x >= 318:
                x = 317
            if y >= 238:
                y = 2
            #print('(x, y): ', x, y)
            people_dist = 0
            for i in [-2,-1, 0, 1, 2]:
                for j in [-2,-1, 0, 1, 2]:
                    if depth_frame_arr[y+i][x+j] > 0:
                        people_dist = people_dist + depth_frame_arr[y+i][x+j]
                        k = k + 1
            #print(p, people_dist)
            if people_dist > 0:
                people_dist = people_dist/k
                #print('people_dist, k: ',people_dist, k)
                if people_dist <= people_dist_min:
                    people_dist_min = people_dist
                    people_index = p
                    #print('people_dist <= people_dist_min: ',p, people_dist)

        #print('(people_index, people_dist_min): ', people_index, people_dist_min)
        head_x = skeleton['people'][people_index]['pose_keypoints_2d'][0]
        head_y = skeleton['people'][people_index]['pose_keypoints_2d'][1]
        L_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][12]
        L_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][13]
        R_hand_x = skeleton['people'][people_index]['pose_keypoints_2d'][21]
        R_hand_y = skeleton['people'][people_index]['pose_keypoints_2d'][22]
        L_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][30]
        L_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][31]
        R_leg_x = skeleton['people'][people_index]['pose_keypoints_2d'][39]
        R_leg_y = skeleton['people'][people_index]['pose_keypoints_2d'][40]

        adjust_hand = 15
        adjust_leg = 15
        head = frame.crop((head_x-24, head_y - 24, head_x + 24, head_y + 24)) #2*3+1
        L_hand = frame.crop((L_hand_x-24, L_hand_y - 24 + adjust_hand, L_hand_x + 24, L_hand_y + 24 + adjust_hand))
        R_hand = frame.crop((R_hand_x-24, R_hand_y - 24 + adjust_hand, R_hand_x + 24, R_hand_y + 24 + adjust_hand))
        L_leg = frame.crop((L_leg_x-24, L_leg_y - 24 + adjust_leg, L_leg_x + 24, L_leg_y + 24 + adjust_leg))
        R_leg = frame.crop((R_leg_x-24, R_leg_y - 24 + adjust_leg, R_leg_x + 24, R_leg_y + 24 + adjust_leg))

        frame_concat=Image.new( 'RGB' , (48,240) , (0,0,0) )
        if flip:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(R_hand, (0,48))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(R_leg, (0,144))
            frame_concat.paste(L_leg, (0,192))
            frame_concat=frame_concat.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,48))
            frame_concat.paste(R_hand, (0,96))
            frame_concat.paste(L_leg, (0,144))
            frame_concat.paste(R_leg, (0,192))
        #print("people number: ", len(skeleton['people']) )
        return frame_concat

    else:
        #print(len(skeleton['people']))
        return ''

'''
Duplicate:  view_1/a01_s08_e00  S08A01E00V01
Duplicate:  view_1/a01_s08_e02  S08A01E02V01
Duplicate:  view_1/a01_s08_e03  S08A01E03V01
interval == 0:  S09A03E02V01
S02A02E04V03
'''

file_count = 0
skeleton_file_name = ''
debug=False

def construct_st_roi(filename, evaluation=False, random_interval=False,random_roi_move=False,random_flip=False, temporal_rgb_frames=5):
    sequence_length = temporal_rgb_frames + 1
    action_id = int(
        filename[filename.find('A') + 1:filename.find('A') + 3])
    subject_id = int(
        filename[filename.find('S') + 1:filename.find('S') + 3])
    environment_id = int(
        filename[filename.find('E') + 1:filename.find('E') + 3])
    view_id = int(
        filename[filename.find('V') + 1:filename.find('V') + 3])
    skeleton_file_name, rgb_sample_name = filename_construct(subject_id, action_id, environment_id, view_id)

    frame_file = openpose_path + skeleton_file_name

    if os.path.isdir(frame_file):# and action_id == 50:

        # load the frames' file name from folder
        frames = os.listdir(frame_file)
        frames_rgb = rgb_frames(frame_path, rgb_sample_name)

        fivefs_concat=Image.new( 'RGB' , (48*temporal_rgb_frames,240) , (0,0,0) )
        if_2_people = False
        i=0
        # checked all len(frames) are  > 6
        sample_interval = len(frames) // sequence_length
        flip = False
        if sample_interval == 0:
            #print('interval == 0: ', skeleton_file_name, len(frames))
            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            #sample_indexs = f(150, seq_info['numFrame'])
            #frame_range = [1,1,2,2,2,3,3]
            frame_range = f(temporal_rgb_frames, len(frames))
        else:
            if not evaluation:
                # Randomly choose sample interval and start frame
                start_i=0
                if random_interval:
                    sample_interval = np.random.randint(1, len(frames) // sequence_length + 1)
                    start_i = np.random.randint(0, len(frames) - sample_interval * sequence_length + 1)
                #if random_roi_move:

                if random_flip:
                    flip = np.random.random() < 0.5
                # aline selection to the two sides

                frame_range = range(start_i, len(frames), sample_interval)
                #print(flip)
                #print(start_i, sample_interval)
            else:
                # Start at first frame and sample uniformly over sequence
                start_i = 0
                flip = False
                frame_range = range(start_i, len(frames), sample_interval)
        #print(frame_range)
        for frame in frame_range:#[1,1,2,2,3]:

            if frame != 0 and frame != (sequence_length*sample_interval):

                if not debug:
                    #openpose_file_, frame_file_ = openposeFile(frame_file, frame, skeleton_file_name, openpose_path)
                    frame_croped = ''
                    frame_ = frame
                    # find the closest non'' frame
                    while frame_croped == '':
                        openpose_file_ = openposeFile(frame_file, frame_, skeleton_file_name, openpose_path)
                        #print(frames_rgb)

                        frame_file_ = frame_path + rgb_sample_name +'/'+ frames_rgb[frame_]
                        # both openpose and RGB frame should exist
                        if os.path.isfile(openpose_file_) and os.path.isfile(frame_file_):
                            frame_croped = cropBody(openpose_file_, frame_file_, action_id, flip)

                            if frame_croped == 2:
                                if_2_people = True
                                frame_croped = Image.new( 'RGB' , (48,240) , (0,0,0) )
                        else:
                            #print('file_unconsistent: ',openpose_file_, os.path.isfile(openpose_file_))
                            string = str(frame_file_ + " {}\n" + openpose_file_ + ' {}\n').format(os.path.isfile(frame_file_), os.path.isfile(openpose_file_))
                            with open('file_unconsistent_crop.txt', 'a') as fd:
                                fd.write(f'\n{string}')
                        #print(openpose_file_, frame_file_)
                        frame_ = frame_ + 1
                        if frame_ > len(frames):
                            frame_croped = Image.new( 'RGB' , (48,240) , (0,0,0) )
                            break

                    fivefs_concat.paste(frame_croped, (i*48+1,0))
                    i+=1
    '''
    fivefs_concat = transform(fivefs_concat)
    if flip:
        fivefs_concat = torch.flip(fivefs_concat, (-1,))
    '''
    return fivefs_concat

if __name__ == '__main__':
    '''
    sss = 1 # setup_id
    aaa = 1 # camera_id
    eee = 1 # subject_id
    vvv = 1 # duplicate_id
    for subject_id in range(1,11):     # 1:20 Diferernt height and distance
        print('Subject: ',subject_id)
        if subject_id < sss:
            continue
        for action_id in range(1,11):     # 1:3 camera views
            if subject_id < sss + 1 and action_id < aaa:
                continue
            for environment_id in range(1,12):   # 1:40 distinct subjects aged between 10 to 35
                if subject_id < sss + 1 and action_id < aaa + 1 and environment_id < eee:
                    continue
                for view_id in range(1,4):  # 1:2 Performance action twice, one to left camera, one to right camera
                    if subject_id < sss + 1 and action_id < aaa + 1 and environment_id < eee + 1 and view_id < vvv:
                        continue

                    skeleton_file_name, rgb_sample_name = filename_construct(subject_id, action_id, environment_id, view_id)
                    frame_file = openpose_path + skeleton_file_name

                    if os.path.isdir(frame_file):
                        fivefs_concat = construct_st_roi(skeleton_file_name, evaluation=True)

                        frames_save = save_path + skeleton_file_name +'.png'
                        fivefs_concat.save(frames_save,"PNG")
    #'''
    fivefs_concat = construct_st_roi('S01A05E01V02', evaluation=False, temporal_rgb_frames=3)
    #fivefs_concat = np.einsum('kli->lik',fivefs_concat.numpy())
    plt.imshow(fivefs_concat)
    plt.suptitle('Corpped Body Parts')
    plt.show()
