
% fprintf(1, 'loading...\n');
% load('iLab20M-data/imagefiles.mat');

nImages = length(imageFilesiLab);
imgDirs = cell(nImages,1);
imgNames = cell(nImages,1);

fprintf(1, 'parsing...\n');
parfor i=1:nImages
    [imgDirs{i},imgNames{i},~] = fileparts(imageFilesiLab{i}); 
    imgNames{i} = [imgNames{i} '.png'];
end

fprintf(1, 'building index...\n');
[u_imgDirs, idx, idx_u]   = unique(imgDirs);

% return;

% nUDirs = length(u_imgDirs);
% imgDirsIdx  = uint8 (zeros(nImages,1));
% parfor i=1:nImages
% %     idx = strfind(u_imgDirs, imgDirs{i});
% %     idx = find(~cellfun('isempty', idx));
%     i_imgDir = imgDirs{i};
%     for k=1:nUDirs
%         if strcmp(i_imgDir, u_imgDirs{k})
%             idx = k; 
%             imgDirsIdx(i) = uint8(idx);
%             break;
%         end
%     end    
% end

imgDirsIdx = idx;
imgDirs = u_imgDirs;
save('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-data/imagefiles-indexed.mat', ...
        'imgNames', 'imgDirs', 'imgDirsIdx', '-v7.3');
    
%% parse each file name    
instancesIdx    = uint8(zeros(nImages,1));   
backgroundsIdx  = uint16(zeros(nImages,1));
lightsIdx       = uint8(zeros(nImages,1));
focusIdx        = uint8(zeros(nImages,1));
camerasIdx      = uint8(zeros(nImages,1));
rotationsIdx    = uint8(zeros(nImages,1));
objectsIdx      = uint8(zeros(nImages,1));

fprintf(1, 'parsing file names...\n');
parfor i=1:nImages
    fileName = imgNames{i};
    options =  parseImgNameiLab(fileName);
    
    instancesIdx(i)     =   uint8(options.instance);
    backgroundsIdx(i)   =   uint16(options.background);
    lightsIdx(i)        =   uint8(options.lighting);
    rotationsIdx(i)     =   uint8(options.rotation);
    camerasIdx(i)       =   uint8(options.camera);
    focusIdx(i)         =   uint8(options.focus);    
    objectsIdx(i)       =   uint8(getClassIdxiLab(options.class));    
    
end

fprintf(1, 'saving...\n');

fid = fopen('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-data/imagefiles-lists.txt', 'wt');
for i=1:nImages
    fprintf(fid, '%s\n', imgNames{i});
end
fclose(fid);

classes = getClassesiLab;
save('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-data/imagefiles-info.mat', ...
        'imgDirs', 'imgDirsIdx', ...
        'instancesIdx', 'backgroundsIdx', 'lightsIdx', ...
        'rotationsIdx', 'camerasIdx',  'focusIdx', 'objectsIdx', 'classes',  '-v7.3');    
    
clear all;    

