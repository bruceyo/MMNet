import os
import glob
from datetime import date
from pathlib import Path

# If the number of bvh file is zero, execute the below code
os.system("sudo chmod 777 /home/irteam/Mocap_AI")
#today = str(date.today()).replace("-","")[2:] + "_skeletons"
today = 'Updated_bvhlist'

if not(os.path.isdir(today)):
   os.mkdir(os.path.join(today))

origin_list = glob.glob('./Completed/*.txt')
findOrigin = []
for ori in origin_list:
  findOrigin.append(ori[-30:-4])

dir_path = "/content/mocap/*/*/*/*/"
all_dir = glob.glob('/home/irteam/Mocap_AI/163. 가구가전사무기기 사용 모션캡처 데이터/1.원천데이터/*/*/*/*/*/*/*.bvh', recursive = True)
bvh = 0
findPair = []
for files in all_dir:
  bvh = bvh + 1
  findPair.append(files[-30:-4])
print('total number of bvh: ', bvh)

all_dir2 = glob.glob('/home/irteam/Mocap_AI/163. 가구가전사무기기 사용 모션캡처 데이터/2.라벨링데이터/*/*/*/*/*/*/*.json', recursive = True)
num = 0
pairnum = 0
tmp = []
result = ""
cnt = 0
f = open(today + "/bvh_list_"+str(cnt)+".txt", 'w')
print("bvh_list_"+str(cnt)+".txt : Open")
for files in all_dir2:
  if files[-31:-5] in findPair and files[-31:-5] not in findOrigin:
    tmp = files.replace('.json', '.bvh')
    tmp = tmp.replace('2.라벨링데이터', '1.원천데이터')
    
    result = result + tmp + "\n"
    pairnum = pairnum + 1
    if pairnum % 1000 == 0:
      f.write(result)   # 기존꺼 닫음
      f.close()
      print("bvh_list_"+str(cnt)+".txt : Done")
      cnt = cnt + 1
      f = open(today + "/bvh_list_"+str(cnt)+".txt", 'w') # 새로운거 열음
      print("bvh_list_"+str(cnt)+".txt : Open")
      result = ""

  num = num + 1

f.write(result)   # 기존꺼 닫음
f.close()
# print("bvh_list_for_rgb_230102".txt : Done")
print('total number of json: ', num)
print('total number of pair: ', pairnum)
