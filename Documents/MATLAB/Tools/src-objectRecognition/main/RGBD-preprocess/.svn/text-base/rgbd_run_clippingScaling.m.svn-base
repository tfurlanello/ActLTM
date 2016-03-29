%% note:
%% this script is problematic, since different pictures of the same 
% instance under the same camera is not scaled by an uniform scale

saveCroppedDir = '/lab/igpu3/u/jiaping/washington-RGBD/dataset/cropped';

%% run clipping
imInfoFile = '/lab/jiaping/svn-jiaping/projects/iLab-object-recognition/src/main/RGBD-data-info/imgInfo-center-scale.mat';
load(imInfoFile);
nImages = numel(imInfo);

for i=1:nImages
   
   if rem(i,500) == 0
       i  %monitoring the progress
   end
   i_imInfo  = imInfo(i);   
   objCenter = imInfo(i).center;
   scale     = imInfo(i).scale;
   [crop, cropbox] = rgbd_cropImg2(i_imInfo, objCenter, scale);
   
   imname = rgbd_genImgFileName(i_imInfo, 'raw');
   cropinfoName = [imname(1:end-4) '.cropbox'];
   
   imwrite(crop, fullfile(saveCroppedDir, imname));
   fid = fopen(fullfile(saveCroppedDir, cropinfoName), 'w');
   fprintf(fid, '%d %d %d %d\n', cropbox(1), cropbox(2), ...
                        cropbox(3), cropbox(4));
   fclose(fid);
   
end


%% run scaling
% rescale each image to [256 256]
saveScaledDir = '/lab/igpu3/u/jiaping/washington-RGBD/dataset/scaled-256x256';
fixedSize = [256 256];
for i=1:nImages
   if rem(i,500) == 0
       i  %monitoring the progress
   end
   i_imInfo = imInfo(i);   
   imname = rgbd_genImgFileName(i_imInfo, 'raw');    
   
   im = imread(fullfile(saveCroppedDir, imname));
   im_scaled = imresize(im, fixedSize);
   
   save_imname = [imname(1:end-4) '.jpg'];
   imwrite(im_scaled, fullfile(saveScaledDir, save_imname));    
end

