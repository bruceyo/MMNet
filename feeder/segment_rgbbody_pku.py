#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 15:23:37 2019

@author: bruce
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import csv
import numpy as np

skeleton_ntu_path = '/media/bruce/2Tssd/data/pku_skeleton_ntu/' # need to move to 2Tssd
frame_path = '/media/bruce/2Tssd/data/pku_rgb_frames/'
label_frame_path = '/media/bruce/2Tssd/data/Train_Label_PKU_final/'
with open(label_frame_path + 'file_frame_dic.csv', mode='r') as infile:
    reader = csv.reader(infile)
    file_frame_dic = {rows[0]:rows[1] for rows in reader}
openpose_path = '/media/bruce/2T/data/openpose_pkummd/'
save_path = '/media/bruce/2Tssd/data/pku_rgb_frames_crop/fivefs/'
debug = False
view_dic = {1:'L',2:'M',3:'R'}

def filename_construct(file_id, view_id, class_id, label_id):
    file_name = ''
    if file_id/100 >= 1:
        file_name = file_name +'0' + str(file_id)
    elif file_id/10 >= 1:
        file_name = file_name +'00' + str(file_id)
    else:
        file_name = file_name + '000' +  str(file_id)

    file_name = file_name + '-' +  view_dic[view_id]

    skeleton_file_name = ''
    if file_id/100 >= 1:
        skeleton_file_name = skeleton_file_name +'F' + str(file_id)
    elif file_id/10 >= 1:
        skeleton_file_name = skeleton_file_name +'F0' + str(file_id)
    else:
        skeleton_file_name = skeleton_file_name + 'F00' +  str(file_id)

    if view_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'V0' +  str(view_id)
    else:
        skeleton_file_name = skeleton_file_name + 'V00' +  str(view_id)
    if class_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'C0' +  str(class_id)
    else:
        skeleton_file_name = skeleton_file_name + 'C00' +  str(class_id)

    if label_id/10 >= 1:
        skeleton_file_name = skeleton_file_name + 'L0' +  str(label_id)
    else:
        skeleton_file_name = skeleton_file_name + 'L00' +  str(label_id)

    return file_name, skeleton_file_name

def openposeFile(avi_frames_path, openpose_frame_path, frame):
    frame_file_ = avi_frames_path + '/' + str(frame) + '.jpg'
    frame_ = '';
    if frame/1000 >= 1:
        frame_ = str(frame)
    elif frame/100 >= 1:
        frame_ = '0' + str(frame)
    elif frame/10 >= 1:
        frame_ = '00' + str(frame)
    else:
        frame_ ='000' + str(frame)
    openpose_file_ = openpose_frame_path + '_00000000'+ frame_ + '_keypoints.json'
                                     #0002-L_000000000000_keypoints
    return openpose_file_, frame_file_

def cropBody(openpose_file, frame_file, action_id):
    #upper=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #lower=Image.new( 'RGB' , (224,112) , (0,0,0) )
    #whole=Image.new( 'RGB' , (224,448) , (0,0,0) )

    frame = Image.open(frame_file)

    if openpose_file:
        with open(openpose_file, 'r') as f:
            skeleton = json.load(f)
    #print(len(skeleton['people']))
    # calculate which people?
    if action_id not in [12, 14, 16, 18, 21, 24, 26, 27]: # or action_id > 49:
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

        head = frame.crop((head_x-48, head_y - 48, head_x + 48, head_y + 48)) #2*3+1
        L_hand = frame.crop((L_hand_x-48, L_hand_y - 48, L_hand_x + 48, L_hand_y + 48))
        R_hand = frame.crop((R_hand_x-48, R_hand_y - 48, R_hand_x + 48, R_hand_y + 48))
        L_leg = frame.crop((L_leg_x-48, L_leg_y - 48, L_leg_x + 48, L_leg_y + 48))
        R_leg = frame.crop((R_leg_x-48, R_leg_y - 48, R_leg_x + 48, R_leg_y + 48))

        frame_concat=Image.new( 'RGB' , (96,480) , (0,0,0) )
        frame_concat.paste(head, (0,0))
        frame_concat.paste(L_hand, (0,96))
        frame_concat.paste(R_hand, (0,192))
        frame_concat.paste(L_leg, (0,288))
        frame_concat.paste(R_leg, (0,384))

        #print('frame_concat   if')
        return frame_concat

    elif len(skeleton['people']) > 0:
        if len(skeleton['people']) > 1:
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

            head = frame.crop((head_x-24, head_y - 48, head_x + 24, head_y + 48)) #2*3+1
            L_hand = frame.crop((L_hand_x-24, L_hand_y - 48, L_hand_x + 24, L_hand_y + 48))
            R_hand = frame.crop((R_hand_x-24, R_hand_y - 48, R_hand_x + 24, R_hand_y + 48))
            L_leg = frame.crop((L_leg_x-24, L_leg_y - 48, L_leg_x + 24, L_leg_y + 48))
            R_leg = frame.crop((R_leg_x-24, R_leg_y - 48, R_leg_x + 24, R_leg_y + 48))

            head_x_1 = skeleton['people'][1]['pose_keypoints_2d'][0]
            head_y_1 = skeleton['people'][1]['pose_keypoints_2d'][1]
            L_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][12]
            L_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][13]
            R_hand_x_1 = skeleton['people'][1]['pose_keypoints_2d'][21]
            R_hand_y_1 = skeleton['people'][1]['pose_keypoints_2d'][22]
            L_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][30]
            L_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][31]
            R_leg_x_1 = skeleton['people'][1]['pose_keypoints_2d'][39]
            R_leg_y_1 = skeleton['people'][1]['pose_keypoints_2d'][40]

            head_1 = frame.crop((head_x_1-24, head_y_1 - 48, head_x_1 + 24, head_y_1 + 48)) #2*3+1
            L_hand_1 = frame.crop((L_hand_x_1-24, L_hand_y_1 - 48, L_hand_x_1 + 24, L_hand_y_1 + 48))
            R_hand_1 = frame.crop((R_hand_x_1-24, R_hand_y_1 - 48, R_hand_x_1 + 24, R_hand_y_1 + 48))
            L_leg_1 = frame.crop((L_leg_x_1-24, L_leg_y_1 - 48, L_leg_x_1 + 24, L_leg_y_1 + 48))
            R_leg_1 = frame.crop((R_leg_x_1-24, R_leg_y_1 - 48, R_leg_x_1 + 24, R_leg_y_1 + 48))

            frame_concat=Image.new( 'RGB' , (96,480) , (0,0,0) )
            frame_concat.paste(head, (0,0))
            frame_concat.paste(head_1, (48,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(L_hand_1, (48,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(R_hand_1, (48,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(L_leg_1, (48,288))
            frame_concat.paste(R_leg, (0,384))
            frame_concat.paste(R_leg_1, (48,384))
            #print('frame_concat   elif')
            return frame_concat
        else:
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

            head = frame.crop((head_x-48, head_y - 48, head_x + 48, head_y + 48)) #2*3+1
            L_hand = frame.crop((L_hand_x-48, L_hand_y - 48, L_hand_x + 48, L_hand_y + 48))
            R_hand = frame.crop((R_hand_x-48, R_hand_y - 48, R_hand_x + 48, R_hand_y + 48))
            L_leg = frame.crop((L_leg_x-48, L_leg_y - 48, L_leg_x + 48, L_leg_y + 48))
            R_leg = frame.crop((R_leg_x-48, R_leg_y - 48, R_leg_x + 48, R_leg_y + 48))

            frame_concat=Image.new( 'RGB' , (96,480) , (0,0,0) )
            frame_concat.paste(head, (0,0))
            frame_concat.paste(L_hand, (0,96))
            frame_concat.paste(R_hand, (0,192))
            frame_concat.paste(L_leg, (0,288))
            frame_concat.paste(R_leg, (0,384))

            #print('frame_concat   if')
            return frame_concat
    else:
        #print(len(skeleton['people']))
        return ''

negelect_list = [
    'F089V001C023L001',
    'F089V002C023L001',
    'F089V003C023L001',
    'F043V001C001L021',
    'F133V002C033L013',
    'F241V003C001L025'
    ]
# to set the continue point S006C002P019R002A039
fff = 2 # file_id
vvv = 1 # view_id
ccc = 1 # class_id
lll = 1 # label_id

file_count = 0
skeleton_file_name = ''
sequence_length = 6

done = False
'''
for file_id in range(2,365):     # 2:364 Diferernt height and distance
    if file_id < fff:
        continue
    for view_id in range(1,4):     # 1:3 camera views
        if file_id < fff + 1 and view_id < vvv:
            continue
        for class_id in range(1,52):   # 1:51 distinct subjects aged between 10 to 35
            if file_id < fff + 1 and view_id < vvv + 1 and class_id < ccc:
                continue
            for label_id in range(1,29):  # 1:28 Performance action twice, one to left camera, one to right camera
                if file_id < fff + 1 and view_id < vvv + 1 and class_id < ccc + 1 and label_id < lll:
                    continue
'''
def construct_st_roi(filename, evaluation=False, random_interval=False,random_roi_move=False,random_flip=False, temporal_rgb_frames=5):
    sequence_length = temporal_rgb_frames + 1
    file_id = int(
        filename[filename.find('F') + 1:filename.find('F') + 4])
    view_id = int(
        filename[filename.find('V') + 1:filename.find('V') + 4])
    class_id = int(
        filename[filename.find('C') + 1:filename.find('C') + 4])
    label_id = int(
        filename[filename.find('L') + 1:filename.find('L') + 4])
    #print('file_id: {}, view_id: {}, class_id: {}, label_id: {}'.format(file_id, view_id, class_id, label_id))
    file_name, skeleton_file_name = filename_construct(file_id, view_id, class_id, label_id)

    if skeleton_file_name in negelect_list:
        return ''
    skeleton_ntu_file_name = skeleton_ntu_path + skeleton_file_name + '.skeleton'
    fivefs_concat=Image.new( 'RGB' , (96*temporal_rgb_frames,480) , (0,0,0) )
    if os.path.isfile(skeleton_ntu_file_name):

        avi_frames_path = frame_path + file_name
        openpose_frame_path = openpose_path + file_name + '.avi/' + file_name
        frames = file_frame_dic[skeleton_file_name].split('_')
        frames[0] = int(float(frames[0].strip()))
        frames[1] = int(float(frames[1].strip()))
        frames_len = frames[1] - frames[0]
        start_i = frames[0]
        # checked all frames_len are  > 6
        sample_interval = frames_len // sequence_length
        flip = False

        if sample_interval == 0:
            f = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
            frame_range = f(temporal_rgb_frames, len(frames))
        else:
            if not evaluation:
                # Randomly choose sample interval and start frame
                #start_i=0
                if random_interval:
                    #print('random_interval:::::::::::::',random_interval)
                    sample_interval = np.random.randint(1, frames_len // sequence_length + 1)
                    start_i = start_i + np.random.randint(0, frames_len - sample_interval * sequence_length + 1)
                #if random_roi_move:

                if random_flip:
                    flip = np.random.random() < 0.5

                # aline selection to the two sides

                frame_range = range(start_i, frames[1] + 1, sample_interval)
                #print(flip)
                #print(start_i, sample_interval)
            else:
                # Start at first frame and sample uniformly over sequence
                #start_i = 0
                flip = False
                frame_range = range(start_i, frames[1] + 1, sample_interval)

        i = 0
        for frame in frame_range:#range(start_i, frames[1] + 1, sample_interval):
            #print(frame)
            if frame != 0 and frame != (sequence_length*sample_interval):
                #print(frame)
                if not debug:
                    #openpose_file_, frame_file_ = openposeFile(frame_file, frame, skeleton_file_name, openpose_path)
                    frame_croped = ''
                    frame_ = frame
                    # find the closest non'' frame
                    while frame_croped == '':
                        openpose_file_, frame_file_ = openposeFile(avi_frames_path, openpose_frame_path, frame_)
                        # both openpose and RGB frame should exist
                        #print(openpose_file_, os.path.isfile(openpose_file_))
                        if os.path.isfile(openpose_file_) and os.path.isfile(frame_file_):
                            frame_croped = cropBody(openpose_file_, frame_file_, class_id)
                            #print('file consistent: ',openpose_file_)
                        else:
                            #print('file_unconsistent: ',openpose_file_, os.path.isfile(openpose_file_))
                            string = str(frame_file_ + " {}\n" + openpose_file_ + ' {}\n').format(os.path.isfile(frame_file_), os.path.isfile(openpose_file_))
                            with open('file_unconsistent_crop.txt', 'a') as fd:
                                fd.write('\n'+string)

                        frame_ = frame_ + 1
                        if frame_ > frames[1] + 1:
                            frame_croped = Image.new( 'RGB' , (96,480) , (0,0,0) )
                            string = str("Empty: "+ frame_file_ + " {}\n" + openpose_file_ + ' {}\n').format(os.path.isfile(frame_file_), os.path.isfile(openpose_file_))
                            with open('file_unconsistent_crop.txt', 'a') as fd:
                                fd.write('\n'+string)
                            break

                    fivefs_concat.paste(frame_croped, (i*96+1,0))
                    i+=1

    return fivefs_concat

if __name__ == '__main__':
    '''
    import os
    import json
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from PIL import Image
    import numpy as np

    frame_path = '../../data/NWestern_UCLA/multiview_action/'
    openpose_path = '../../data/NWestern_UCLA/openpose_ucla/'
    save_path = '../../data/NWestern_UCLA/ucla_rgb_frames_crop/fivefs_240/'
    debug = False
    duplicate_list=[]
    frame_croped = transform(frame_croped)
    frame_croped = torch.flip(frame_croped, (-1,))
    frame_croped = np.einsum('kli->lik',frame_croped.numpy())
    '''

    fivefs_concat_ = construct_st_roi('F282V003C051L001', evaluation=False,random_interval=False, random_flip=True)
    #fivefs_concat = np.einsum('kli->lik',fivefs_concat.numpy())
    plt.imshow(fivefs_concat_)
    plt.suptitle('Corpped Body Parts')
    plt.show()
