#!/bin/bash
cd /media/bruce/2T/models/openpose-master

rgb_file_name_pre="/mnt/nas/ntu-rgbd/PKUMMD/Data/RGB_VIDEO/"
output_path_pre="/media/bruce/2T/data/openpose_pkummd/"

video=10
#views=("L" "M" "R")
views='L M R'
count=0
#Declare a string array

#smb://158.132.255.124/ntu-rgbd/Northwestern-UCLA/multiview_action_videos/a01/v01_s01_e00.avi
for video_id in $(seq 1 1 364); #12
do
  if [ $video_id -lt $video ]
  then
    continue
  fi # Voox is done
  for view_id in $views;#${views[@]};
  do
    #echo $view_id

  	#environment_id=$(($environment_id-1))

  	#construct the file name start
  	if [ $(($video_id/100)) -ge 1 ]
  	then
    	rgb_file_name=$rgb_file_name_pre"0"$video_id"-"$view_id".avi" #"a01/v01_s01_e00.avi"
    	output_path=$output_path_pre"0"$video_id"-"$view_id".avi"
    elif [ $(($video_id/10)) -ge 1 ]
    then
      rgb_file_name=$rgb_file_name_pre"00"$video_id"-"$view_id".avi"
    	output_path=$output_path_pre"00"$video_id"-"$view_id".avi"
  	else
    	rgb_file_name=$rgb_file_name_pre"000"$video_id"-"$view_id".avi"
    	output_path=$output_path_pre"000"$video_id"-"$view_id".avi"
  	fi

  	#construct the file name end

  	#do retrival start
  	if [ -e $rgb_file_name ]
  	then
      count=$(($count+1))
    	echo $(date)" Retrieve Start: "$rgb_file_name >> "/media/bruce/2T/data/openpose_ucla_retrieve.log"
    	./build/examples/tutorial_wrapper/6_user_asynchronous_output.bin --video $rgb_file_name --write_json $output_path --logging_level 3 --face true --hand true --hand_tracking true --num_gpu 1 --num_gpu_start 1
    	echo $(date)" Retrieve End: " $output_path >> "/media/bruce/2T/data/openpose_ucla_retrieve.log"
  	else
       #echo "video_id: "$video_id$rgb_file_name
  	   echo $(date)" File doesn't exist: "$rgb_file_name >> "/media/bruce/2T/data/openpose_ucla_retrieve.log"
       count=$(($count))
    fi
  	#do retrival end



  done
done
echo $count

#./build/examples/tutorial_wrapper/6_user_asynchronous_output.bin --video /mnt/nas/ntu-rgbd/NTU/RGB_videos/nturgb+d_rgb/S001C001P001R001A001_rgb.avi --write_json /media/bruce/2T/data/openpos
