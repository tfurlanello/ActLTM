
% lights invariance training and test data setting file
lightsInvarianceData = ...
    load('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-experiments/data-lightsInvariance.mat');


iLabInfo = load('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-data-info/imagefiles-info.mat');
fid         = fopen('/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-data-info/imagefiles-lists.txt', 'r');
imageNames  = textscan(fid, '%s\n');
imageNames = imageNames{1};
fclose(fid);


%% training images
bTrain  =  lightsInvarianceData.bTrain;

imageNames_train = imageNames(bTrain);
imgDirsIdx_train = iLabInfo.imgDirsIdx(bTrain);
nImages_train   =   sum(bTrain);
labels_train    =   lightsInvarianceData.labelsTrain;

% move the images to a new place
saveDir = '/lab/tmp2ig2/u/jiaping/lightsInvariance';

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
        for k=1:numel(labels_train)
            fprintf(fid, '%s\n', labels_train{k});
        end
        fclose(fid);    
    end
    
end


%% test images
labels_test     =   lightsInvarianceData.labelsTest;
bTest           =  lightsInvarianceData.bTest;
imageNames_test = imageNames(bTest);
imgDirsIdx_test = iLabInfo.imgDirsIdx(bTest);
nImages_test    =   sum(bTest);


% move the images to a new place
saveDir = '/lab/tmp2ig2/u/jiaping/lightsInvariance';

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
        for k=1:numel(labels_test)
            fprintf(fid, '%s\n', labels_test{k});
        end
        fclose(fid);        
    end    
    
end





 