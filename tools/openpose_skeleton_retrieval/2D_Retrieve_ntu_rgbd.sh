#!/bin/bash
cd /media/bruce/2T/models/openpose-master

rgb_file_name_pre=".../ntu-rgbd/NTU/RGB_videos/nturgb+d_rgb/"
output_path_pre="./data/openpose/"
sss=18
ccc=1
ppp=8
rrr=1
aaa=61
#S018C001P008R001A061
for setup_id in $(seq 1 1 20); #20
do
  echo "setup_id"
  if [ $setup_id -lt $sss ]
  then
    continue
  fi # soox is done
  for camera_id in $(seq 1 1 3); #3
  do
    if ([ $setup_id -lt $(($sss+1)) ] && [ $camera_id -lt $ccc ])
    then
      continue
    fi # Coox is done
    for subject_id in $(seq 1 1 40); #40
    do
      if ([ $setup_id -lt $(($sss+1)) ] && [ $camera_id -lt $(($ccc+1)) ] && [ $subject_id -lt $ppp ])
      then
        continue
      fi # Poox is done
      for duplicate_id in $(seq 1 1 2); #2
      do
        if ([ $setup_id -lt $(($sss+1)) ] && [ $camera_id -lt $(($ccc+1)) ] && [ $subject_id -lt $(($ppp+1)) ] && [ $duplicate_id -lt $rrr ])
        then
          continue
        fi # Poox is done
        for action_id in $(seq 1 1 60); #60
        do
          if ([ $setup_id -lt $(($sss+1)) ] && [ $camera_id -lt $(($ccc+1)) ] && [ $subject_id -lt $(($ppp+1)) ] && [ $duplicate_id -lt $(($rrr+1)) ] && [ $action_id -lt $aaa ])
          then
            continue
          fi # Poox is done

          #construct the file name start
          if [ $(($setup_id/10)) -ge 1 ]
          then
            rgb_file_name=$rgb_file_name_pre"S0"$setup_id"C00"$camera_id #"P001R001A001_rgb.avi"
            output_path=$output_path_pre"S0"$setup_id"C00"$camera_id #"P001R001A001"
          else
            rgb_file_name=$rgb_file_name_pre"S00"$setup_id"C00"$camera_id #"P001R001A001_rgb.avi"
            output_path=$output_path_pre"S00"$setup_id"C00"$camera_id #"P001R001A001"
          fi
          if [ $(($subject_id/10)) -ge 1 ]
          then
            rgb_file_name=$rgb_file_name"P0"$subject_id"R00"$duplicate_id
            output_path=$output_path"P0"$subject_id"R00"$duplicate_id
          else
            rgb_file_name=$rgb_file_name"P00"$subject_id"R00"$duplicate_id
            output_path=$output_path"P00"$subject_id"R00"$duplicate_id
          fi
          if [ $(($action_id/10)) -ge 1 ]
          then
            rgb_file_name=$rgb_file_name"A0"$action_id"_rgb.avi"
            output_path=$output_path"A0"$action_id
          else
            rgb_file_name=$rgb_file_name"A00"$action_id"_rgb.avi"
            output_path=$output_path"A00"$action_id
          fi
          #construct the file name end

          #do retrival start
          if [ -e $rgb_file_name ]
          then
            echo $(date)" Retrieve Start: "$rgb_file_name >> "./data/openpose_retrieve.log"
            ./build/examples/tutorial_wrapper/6_user_asynchronous_output.bin --video $rgb_file_name --write_json $output_path --logging_level 3 --face true --hand true --hand_tracking true --num_gpu 1 --num_gpu_start 1
            echo $(date)" Retrieve End: " $output_path >> "./data/openpose_retrieve.log"
          else
            echo $(date)" File doesn't exist: "$rgb_file_name >> "./data/openpose_retrieve.log"
          fi
          #do retrival end
        done
      done
    done
  done
done
