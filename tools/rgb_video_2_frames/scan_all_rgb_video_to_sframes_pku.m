framestart = 1;
frameend = 1;
frame_count = 0;

% to set the continue point

fff = 1; % setup_id
ccc = 1; % camera_id


file_count = 0;

%%{
% 0191-L, 0192-L are empty
for file_id = 2:364            % 2:364 Diferernt height and distance
    if file_id < fff
        continue
    end
for view_id = 1:1              % 1:3 camera views
    if file_id < fff + 1 && view_id < ccc
        continue
    end
video_file_name = '';
if file_id/10 < 1
    video_file_name = strcat(strcat(video_file_name,'000'), int2str(file_id));
elseif file_id/100 < 1
    video_file_name = strcat(strcat(video_file_name,'00'), int2str(file_id));
else
    video_file_name = strcat(strcat(video_file_name,'0'), int2str(file_id));
end
if view_id ==1
    video_file_name = strcat(video_file_name,'-L');
elseif view_id ==2
    video_file_name = strcat(video_file_name,'-M');
else
    video_file_name = strcat(video_file_name,'-R');
end

%save...
file_name_to_save = video_file_name;
video_file_name = strcat('Y:\PKUMMD\Data\RGB_VIDEO\',video_file_name);
video_file_name = strcat(video_file_name,'.avi');
if exist(video_file_name, 'file') == 2
    file_count = file_count + 1;
    rgb_video = VideoReader(video_file_name);
    numberOfFrames = rgb_video.NumberOfFrames;
    action_folder = strcat('Y:\PKUMMD\Data\rgb_frames\',file_name_to_save);
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
    disp(strcat(video_file_name, ' not exist!'));
end
skeleton_file_name = '';
if mod(file_count,10) == 0 && file_count ~= 0
    disp(strcat('file_count: ',num2str(file_count)));
    %disp(strcat(strcat('frame_count: ',num2str(frame_count)),strcat(', file_count: ',num2str(file_count))));
    %disp(strcat(strcat('framestart: ',num2str(framestart)),strcat(', frameend: ',num2str(frameend))));
end

end
end
