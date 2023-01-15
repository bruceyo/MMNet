import json
import os
from datetime import date

#today = str(date.today()).replace("-","")[2:] + "_skeletons"
today = 'Updated_bvhlist'
out_path = '/home/irteam/YJ2/' + today

loc_file= '/home/irteam/YJ2/' + today + '/bvh_list_1.txt'

with open(loc_file, "r", encoding="utf8") as bvh_list:
    for bvh in bvh_list:
        bvh=bvh.replace('\n',"")
        bvh = "\'"+bvh + "\'"
        try:
            os.system("python bvh-converter/bvh_converter/__main__.py " + bvh)
        except:  
            os.system("sudo chmod 777 /home/irteam/Mocap_AI")
            os.system("python bvh-converter/bvh_converter/__main__.py " + bvh)
bvh_list.close()