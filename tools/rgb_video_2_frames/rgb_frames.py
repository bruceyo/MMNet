import os
import numpy as np
import argparse
import cv2
from datetime import date
import json


# video dir_path – rgb 데이터 추출 결과를 저장하는 경로
save_dir_path = './Completed_rgb/'

# bvh list dir_path – 추출 할 데이터셋 경로 및 이름 저장 list 파일 이름
tmp= "json_list_for_rgb_230102.txt"

videoErrorCnt = 0
already_result = 0
no_video = 0
not_open = 0
no_json = 0
no_bvh = 0

# 데이터셋 저장 경로 관리자 권한 접근
#os.system("sudo chmod 777 /home/irteam/Mocap_AI/")

if __name__ == '__main__':
    
    # debug
    with open(tmp,'r',encoding='utf8') as bvh_list:

        for file in bvh_list:

            #mp4 파일을 로드
            file = file.replace('\n',"") #필수
            video_file = file.replace('.json','_F.mp4')
            video_file = video_file.replace('2.라벨링데이터','1.원천데이터')

            #구간 태깅 데이터 확인
            
            json_file = file
            # print(json_file)

            #txt 데이터의 이름을 불러와 폴더 생성!
            #폴더가 이미 있는데 
            if os.path.exists(save_dir_path+ json_file[-31:-5]  + '/') :

                #데이터가 있을 경우 continue
                if os.listdir(save_dir_path+ json_file[-31:-5]  + '/') :
                    #근데 그 데이터가 20개가 넘을 경우 skip   

                    already_result = already_result + 1
                    continue
            else : 
                os.makedirs(save_dir_path+ json_file[-31:-5]  + '/')

            try:
                if os.path.isfile(json_file):
                    with open(json_file,'r',encoding='utf8') as j:
                        try:
                            json_data=json.load(j)
                            print(json)
                            j.close()
                            
                        except:
                            no_json = no_json + 1
                            print("json 오류 ")

                    json_data_anno=json_data['annotation']['actionAnnotationList']

                    # 구간태깅
                    start_frame = json_data_anno[1]['start_frame']
                    last_frame = json_data_anno[1]['end_frame']
                    print(start_frame)
                    print(last_frame)
                else:
                    no_json = no_json + 1
                    print("json 오류")
            except:
                no_bvh = no_bvh +1
                print("bvh 오류")

            # csv read
            print(video_file)

            if os.path.isfile(video_file) == False:
                print("영상 데이터 유무 오류")
                no_video = no_video + 1

            #video 파일 열기
            # get file path for desired video and where to save frames locally
            
            # print(video_file)
            cap = cv2.VideoCapture(video_file)

            #비디오 데이터를 열수 없음
            if (cap.isOpened() == False):
                print("video is not exist")
                no_video = no_video + 1
                    

            path_to_save = os.path.abspath(save_dir_path + json_file[-31:-5]  + '/')

            current_frame = 1

            # cap opened successfully
            while (cap.isOpened()):

                # capture each frame
                ret, frame = cap.read()

                if (ret == True):
                    # keep track of how many images you end up with
                    current_frame += 1

                    if(current_frame < start_frame):
                        continue
                    elif(current_frame > last_frame):
                        break

                    # Save frame as a jpg file
                    elif (current_frame % 10 == 0):
                        #print(json_file[:-4])
                        name = json_file[-31:-5] +'_' + str(current_frame) + '.jpg'
                        # print(f'Creating: {name}')
                        cv2.imwrite(os.path.join(path_to_save, name), frame)
                else:
                    not_open = not_open + 1
                    break

            # release capture
            cap.release()
         
            print('bvh error')
            print(no_bvh)

            print('rgb open error')
            print(not_open)

            print('json error')
            print(no_json)

            print('already preprocessed rgb data')
            print(already_result)

            print('unknown error')
            videoErrorCnt = 0

    print('done')