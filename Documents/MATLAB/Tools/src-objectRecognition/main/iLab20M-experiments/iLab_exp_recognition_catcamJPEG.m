
% lights invariance training and test data setting file
catcamData = ...
    load('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-experiments/data-category-camera.mat');


iLabInfo    = iLab_getImgParameters;
imageNames  = iLab_getImgFileNames;


%% training images
bTrain  =  catcamData.bTrain;

imageNames_train = imageNames(bTrain);
imgDirsIdx_train = iLabInfo.imgDirsIdx(bTrain);
nImages_train   =   sum(bTrain);
labelsCatTrain    =   catcamData.labelsCatTrain(bTrain);
labelsCamTrain  = catcamData.labelsCamTrain(bTrain);

trainImagesInfo = struct('backgroundIdx', {}, ...
                         'objectIdx', {}, ...
                         'instanceIdx', {}, ...
                         'cameraIdx', {}, ...
                         'rotationIdx', {}, ...
                         'focusIdx', {}, ...
                         'lightIdx', {});
 cnt = 0;                    
for i=1:numel(bTrain)
    if bTrain(i) == false
        continue;
    end
    cnt = cnt + 1;
    trainImagesInfo(cnt).backgroundIdx = iLabInfo.backgroundsIdx(i);
    trainImagesInfo(cnt).objectIdx     = iLabInfo.objectsIdx(i);
    trainImagesInfo(cnt).instanceIdx   = iLabInfo.instancesIdx(i);
    trainImagesInfo(cnt).cameraIdx     = iLabInfo.camerasIdx(i);
    trainImagesInfo(cnt).rotationIdx   = iLabInfo.rotationsIdx(i);
    trainImagesInfo(cnt).focusIdx      = iLabInfo.focusIdx(i);
    trainImagesInfo(cnt).lightIdx      = iLabInfo.lightsIdx(i);
    
end

% move the images to a new place
saveDir = '/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

intervals = linspace(1, nImages_train, 5);

for i=1:4
    
    if ~exist(fullfile(saveDir, [num2str(i) '-train.txt']), 'file')
        fid = fopen(fullfile(saveDir, [num2str(i) '-train.txt']), 'wt');
        fprintf(fid, '1');
        fclose(fid);
         i
        parfor j=round(intervals(i)):round(intervals(i+1))
             im = imread(fullfile(iLabInfo.imgDirs{imgDirsIdx_train(j)}, imageNames_train{j}));
             imwrite(im, fullfile(saveDir, 'trainImages', [imageNames_train{j}(1:end-4) '.jpg']), 'jpg');      
        end        

    end
    
    if ~exist(fullfile(saveDir, 'metadata', 'trainImageLists.flag'), 'file')
        fid = fopen(fullfile(saveDir, 'metadata', 'trainImageLists.flag'), 'wt');
        fprintf(fid, '1');
        fclose(fid);          
        
        fid = fopen(fullfile(saveDir, 'metadata', 'trainImageLists.txt'), 'wt');
        for k=1:numel(imageNames_train)
            fprintf(fid, '%s\n', [imageNames_train{k}(1:end-4) '.jpg']);
        end
        fclose(fid); 
    end
    
    if ~exist(fullfile(saveDir, 'metadata', 'trainImageLabels.flag'), 'file')
        fid = fopen(fullfile(saveDir, 'metadata', 'trainImageLabels.flag'), 'wt');
        fprintf(fid, '1');
        fclose(fid);          
        
        fid = fopen(fullfile(saveDir, 'metadata', 'trainImageLabels.txt'), 'wt');
        for k=1:numel(labelsCatTrain)
            fprintf(fid, '%s %s\n', labelsCatTrain{k},labelsCamTrain{k});
        end
        fclose(fid);   
        
        save(fullfile(saveDir, 'metadata', 'trainImagesInfo.mat'), 'trainImagesInfo');
    end
    
end


%% test images

bTest           =  catcamData.bTest;
imageNames_test = imageNames(bTest);
imgDirsIdx_test = iLabInfo.imgDirsIdx(bTest);
nImages_test    =   sum(bTest);
labelsCatTest     =   catcamData.labelsCatTest(bTest);
labelsCamTest     =   catcamData.labelsCamTest(bTest);


testImagesInfo = struct('backgroundIdx', {}, ...
                         'objectIdx', {}, ...
                         'instanceIdx', {}, ...
                         'cameraIdx', {}, ...
                         'rotationIdx', {}, ...
                         'focusIdx', {}, ...
                         'lightIdx', {});
cnt = 0;                     
for i=1:numel(bTest)
    if bTest(i) == false
        continue;
    end
    cnt = cnt + 1;
    testImagesInfo(cnt).backgroundIdx = iLabInfo.backgroundsIdx(i);
    testImagesInfo(cnt).objectIdx     = iLabInfo.objectsIdx(i);
    testImagesInfo(cnt).instanceIdx   = iLabInfo.instancesIdx(i);
    testImagesInfo(cnt).cameraIdx     = iLabInfo.camerasIdx(i);
    testImagesInfo(cnt).rotationIdx   = iLabInfo.rotationsIdx(i);
    testImagesInfo(cnt).focusIdx      = iLabInfo.focusIdx(i);
    testImagesInfo(cnt).lightIdx      = iLabInfo.lightsIdx(i);
    
end


intervals = linspace(1, nImages_test, 5);

for i=1:4
    
    if ~exist(fullfile(saveDir, [num2str(i) '-test.txt']), 'file')
        fid = fopen(fullfile(saveDir, [num2str(i) '-test.txt']), 'wt');
        fprintf(fid, '1');
        fclose(fid);
        i
        parfor j=round(intervals(i)):round(intervals(i+1))
             im = imread(fullfile(iLabInfo.imgDirs{imgDirsIdx_test(j)}, imageNames_test{j}));
             imwrite(im, fullfile(saveDir, 'testImages', [imageNames_test{j}(1:end-4) '.jpg']), 'jpg');      
        end
   
    end
    
    
    if ~exist(fullfile(saveDir, 'metadata', 'testImageLists.flag'), 'file')
        fid = fopen(fullfile(saveDir, 'metadata', 'testImageLists.flag'), 'wt');
        fprintf(fid, '1');
        fclose(fid);        
        
        fid = fopen(fullfile(saveDir, 'metadata', 'testImageLists.txt'), 'wt');
        for k=1:numel(imageNames_test)
            fprintf(fid, '%s\n', [imageNames_test{k}(1:end-4) '.jpg']);
        end
        fclose(fid); 
    end

    if ~exist(fullfile(saveDir, 'metadata', 'testImageLabels.flag'), 'file')
        fid = fopen(fullfile(saveDir, 'metadata', 'testImageLabels.flag'), 'wt');
        fprintf(fid, '1');
        fclose(fid);  
        
        fid = fopen(fullfile(saveDir, 'metadata', 'testImageLabels.txt'), 'wt');
        for k=1:numel(labelsCatTest)
            fprintf(fid, '%s %s\n', labelsCatTest{k}, labelsCamTest{k});
        end
        fclose(fid);    
        
        save(fullfile(saveDir, 'metadata', 'testImagesInfo.mat'), 'testImagesInfo');

    end    
    
end

