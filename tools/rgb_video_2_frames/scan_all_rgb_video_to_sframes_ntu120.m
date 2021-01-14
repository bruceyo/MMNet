framestart = 1;
frameend = 1;
frame_count = 0;

% to set the continue point

sss = 18; % setup_id
ccc = 1; % camera_id
ppp = 1; % subject_id
rrr = 1; % duplicate_id
aaa = 61;% action_id

file_count = 0;

%%{
skeleton_file_name = '';
for setup_id = 18:32     % 18:32 Diferernt height and distance
    if setup_id < sss
        continue
    end
for camera_id = 1:3     % 1:3 camera views
    if setup_id < sss + 1 && camera_id < ccc
        continue
    end
for subject_id = 6:106   % 6:106 distinct subjects aged between 10 to 35
    if setup_id < sss + 1 && camera_id < ccc + 1 && subject_id < ppp
        continue
    end
for duplicate_id = 1:2  % 1:2 Performance action twice, one to left camera, one to right camera
    if setup_id < sss + 1 && camera_id < ccc + 1 && subject_id < ppp + 1 && duplicate_id < rrr
        continue
    end
for action_id = 61:120    % 61:120 Action class [11,12,30,31,53]
    if setup_id < sss + 1 && camera_id < ccc + 1 && subject_id < ppp + 1 && duplicate_id < rrr +1 && action_id < aaa
        continue
    end

if setup_id/10 >= 1
    skeleton_file_name = strcat(strcat(skeleton_file_name,'S0'), int2str(setup_id));
else
    skeleton_file_name = strcat(strcat(skeleton_file_name,'S00'), int2str(setup_id));
end
if camera_id/10 >= 1
    skeleton_file_name = strcat(strcat(skeleton_file_name,'C0'), int2str(camera_id));
else
    skeleton_file_name = strcat(strcat(skeleton_file_name,'C00'), int2str(camera_id));
end
if subject_id/100 >= 1
    skeleton_file_name = strcat(strcat(skeleton_file_name,'P'), int2str(subject_id));
elseif subject_id/10 >= 1
    skeleton_file_name = strcat(strcat(skeleton_file_name,'P0'), int2str(subject_id));
else
    skeleton_file_name = strcat(strcat(skeleton_file_name,'P00'), int2str(subject_id));
end
if duplicate_id/10 >= 1
    skeleton_file_name = strcat(strcat(skeleton_file_name,'R0'), int2str(duplicate_id));
else
    skeleton_file_name = strcat(strcat(skeleton_file_name,'R00'), int2str(duplicate_id));
end
if action_id/100 >= 1
    skeleton_file_name = strcat(strcat(skeleton_file_name,'A'), int2str(action_id));
elseif action_id/10 >= 1
    skeleton_file_name = strcat(strcat(skeleton_file_name,'A0'), int2str(action_id));
else
    skeleton_file_name = strcat(strcat(skeleton_file_name,'A00'), int2str(action_id));
end

%save...
file_name_to_save = skeleton_file_name;
skeleton_file_name = strcat('Y:\NTU120\RGB_videos\nturgb+d_rgb\',skeleton_file_name);
skeleton_file_name = strcat(skeleton_file_name,'_rgb.avi');
if exist(skeleton_file_name, 'file') == 2
    file_count = file_count + 1;
    rgb_video = VideoReader(skeleton_file_name);
    numberOfFrames = rgb_video.NumberOfFrames;
    action_folder = strcat('Y:\NTU120\RGB_videos\ntu_rgb_frames\',file_name_to_save);
    if ~exist(action_folder, 'dir')
       mkdir(action_folder);
    end

    for frameNumber = 1:numberOfFrames
        thisFrame = read(rgb_video, frameNumber);

        baseFileName = strcat(num2str(frameNumber-1),'.jpg'); % Whatever....
        fullFileName = fullfile(action_folder, baseFileName);
        pause(0.1)
        imwrite(thisFrame, fullFileName);
        %frame_count = frame_count+1;
    end
else
    %disp(skeleton_file_name);
end
skeleton_file_name = '';
if mod(file_count,500) == 0 && file_count ~= 0
    disp(strcat(', file_count: ',num2str(file_count)));
    %disp(strcat(strcat('frame_count: ',num2str(frame_count)),strcat(', file_count: ',num2str(file_count))));
    %disp(strcat(strcat('framestart: ',num2str(framestart)),strcat(', frameend: ',num2str(frameend))));
end
end
end
end
end
end
