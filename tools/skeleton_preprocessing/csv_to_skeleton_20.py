import json
import pandas as pd
import os 
from datetime import date

# dirname
#today = str(date.today()).replace("-","")[2:] + "_skeletons"
today='update'
#bvh list있는 경로
tmp=today+"/bvh_list_20.txt" # <<수정필요>>

# 저장 폴더경로
out_path='./'+today+'_preprocessed' # skeleton 파일 저장 경로
if not os.path.isdir(out_path):
    os.mkdir(out_path)

#csv 파일 경로
csv_path=today

body_num = "1"
skeleton_num = "25"
joint_name = ["Hips", "Spine1", "Neck", "Head", "LeftArm", "LeftForeArm", "LeftForeArmRoll", "LeftHand", "RightArm", "RightForeArm", "RightForeArmRoll",
              "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "Spine2","rightfinger","thumb_03_r","leftfinger","thumb_03_l"]


with open(tmp,'r',encoding='utf8')as bvh_list: # bvh파일 목록

    for file in bvh_list:    #file : csv파일경로
        
        file=file.replace('\n',"") #필수
        csv_file=file[-30:-4]+'_worldpos.csv'
        csv_file=os.path.join(csv_path,csv_file)

        json_file=file.replace('1.원천데이터','2.라벨링데이터')
        json_file=json_file[:-4]+'.json'
        try:
            if os.path.isfile(json_file):
                with open(json_file,'r',encoding='utf8') as j:
                    try:
                        json_data=json.load(j)
                        j.close()
                    except:
                        with open("errorlist_4.txt", "a") as error_file:
                            data = file + ": " + "json컬럼에러" + "\n"
                            error_file.write(data)
                        continue

                json_data_anno=json_data['annotation']['actionAnnotationList']
                # 구간태깅
                start_frame=json_data_anno[1]['start_frame']
                last_frame=json_data_anno[1]['end_frame']
            else:
                continue
        except:
            with open("errorlist_4.txt", "a") as error_file:
                data = file + ": " + "json에러" + "\n"
                error_file.write(data)
                continue
            
        # csv read
        print('csvfile',csv_file)
        if os.path.isfile(csv_file) == False:
            with open("errorlist_4.txt", "a") as error_file:
                data = file + ": " + "csv 파일 존재X" + "\n"
                error_file.write(data)
            continue
        csv_data=pd.read_csv(csv_file)

        # csv 열 개수 :232가 아니면 넘겨버리기
        if csv_data.shape[1]!=232:
            with open("errorlist_4.txt", "a") as error_file:
                data = file + ": " + "열에러" + "\n"
                error_file.write(data)
            continue
        # 프레임 개수 에러
        if csv_data.shape[0]<last_frame:
            with open("errorlist_4.txt", "a") as error_file:
                data = file + ": " + "프레임에러" + "\n"
                error_file.write(data)
            continue

        with open(out_path+'/'+file[-30:-4]+".txt","w") as new_f: # file[-31:-5] 부분 수정필요 (csv file명에 맞게)
                        #덮어쓰기를 위한 초기화 부분이다.
            new_f.truncate()
            print("working..."+str(new_f))
            frame_num=int((last_frame-start_frame+1)/10) # 다운샘플링: frame개수
            new_f.write((str(frame_num)) + "\n")
            idx = 0
            print('frame시작/끝',start_frame, last_frame)

            count = 0   # 다운샘플링
            for i in range(start_frame,last_frame+1):
                count += 1  # 다운샘플링
                if count%10 != 0:
                    continue
                new_f.write(body_num + "\n")
                new_f.write("0 0 0 0 0 0 0 0 0 0\n") # body정보 10개
                new_f.write(str(skeleton_num) +"\n")

                for x in joint_name:
                    x_index=x+'.X'
                    y_index=x+'.Y'
                    z_index=x+'.Z'

                    if x=='rightfinger':
                        index03=[csv_data.iloc[i]['index_03_r.X'],csv_data.iloc[i]['index_03_r.Y'],csv_data.iloc[i]['index_03_r.Z']]
                        middle03=[csv_data.iloc[i]['middle_03_r.X'],csv_data.iloc[i]['middle_03_r.Y'],csv_data.iloc[i]['middle_03_r.Z']]
                        ring03=[csv_data.iloc[i]['ring_03_r.X'],csv_data.iloc[i]['ring_03_r.Y'],csv_data.iloc[i]['ring_03_r.Z']]
                        pinky03=[csv_data.iloc[i]['pinky_03_r.X'],csv_data.iloc[i]['pinky_03_r.Y'],csv_data.iloc[i]['pinky_03_r.Z']]
                    
                        finger_avg=[(index03[i]+middle03[i]+ring03[i]+pinky03[i])/4 for i in range(len(index03))]
                        new_f.write(str(finger_avg[0])+" ")
                        new_f.write(str(finger_avg[1])+" ")
                        new_f.write(str(finger_avg[2])+" ")
                        new_f.write("0 0 0 0 0 0 0 0 2\n")

                    elif x=='leftfinger':
                        index03=[csv_data.iloc[i]['index_03_l.X'],csv_data.iloc[i]['index_03_l.Y'],csv_data.iloc[i]['index_03_l.Z']]
                        middle03=[csv_data.iloc[i]['middle_03_l.X'],csv_data.iloc[i]['middle_03_l.Y'],csv_data.iloc[i]['middle_03_l.Z']]
                        ring03=[csv_data.iloc[i]['ring_03_l.X'],csv_data.iloc[i]['ring_03_l.Y'],csv_data.iloc[i]['ring_03_l.Z']]
                        pinky03=[csv_data.iloc[i]['pinky_03_l.X'],csv_data.iloc[i]['pinky_03_l.Y'],csv_data.iloc[i]['pinky_03_l.Z']]
                        finger_avg=[(index03[i]+middle03[i]+ring03[i]+pinky03[i])/4 for i in range(len(index03))]
                        new_f.write(str(finger_avg[0])+" ")
                        new_f.write(str(finger_avg[1])+" ")
                        new_f.write(str(finger_avg[2])+" ")
                        new_f.write("0 0 0 0 0 0 0 0 2\n")

                    elif x!='rightfinger' and x!='leftfinger':
                        new_f.write(str(csv_data.iloc[i][x_index]) + " ")
                        new_f.write(str(csv_data.iloc[i][y_index]) + " ")
                        new_f.write(str(csv_data.iloc[i][z_index]) + " ")
                        new_f.write("0 0 0 0 0 0 0 0 2\n")

bvh_list.close()
