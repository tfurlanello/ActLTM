% plot classification results

%% load model
config;
global cnnModel;
model = load(cnnModel) ;
refLabels = model.classes.description;


%% classify test images using CNN
load('testFileInfo.mat');


%% load classification results
saveDir = '/lab/ilab/30/jiaping/iLab20M';

classNames = testFileInfo.classNames;
nClasses = numel(classNames);
classInfo = testFileInfo.classInfo;

visualizationLists = {'plane-0084', 'plane-0083', 'plane-0082', ...
                      'equip-0028', 'equip-0031','equip-0032', ...
                      'equip-0043', 'van-0010' ,'van-0011', 'van-0012', ...
                      'monster-0007', 'monster-0010', 'heli-0001', 'heli-0003', ...
                      'heli-0015', 'train-0005', 'train-0011', 'train-0025', ...
                      'tank-0001', 'tank-0004', 'tank-0023', ...
                      'bus-0002', 'bus-0009', 'bus-0010', ...
                      'boat-0009', 'boat-0010', 'boat-0019', ...
                      'semi-0002', 'semi-0009', 'semi-0018'};
% while 1
rng('shuffle');
rclassIdx = randperm(nClasses);

for ii=1:nClasses
    i = rclassIdx(ii);
    c_instanceInfo  = classInfo(i).instanceInfo;
    c_instanceNames = classInfo(i).instanceNames;
    nInstances      = numel(c_instanceNames);
    rinstanceIdx    = randperm(nInstances);
    for jj=1:nInstances
        j = rinstanceIdx(jj);
        j_instanceName  = c_instanceInfo(j).instanceName;
        imageFiles      = c_instanceInfo(j).instanceFiles;
        nimgs           = numel(imageFiles);
        imageNames      = {};
        
        if sum(cell2mat(strfind(visualizationLists, j_instanceName))) ~= 1
            continue;
        end
        
        resFile = fullfile(saveDir, classNames{i}, j_instanceName, 'predictedLabels.mat');
        if ~exist(resFile, 'file')
            continue;
        end
        
        for k=1:nimgs
            slashIdx = strfind(imageFiles{k}, '/');
            imName = imageFiles{k}((slashIdx(end)+1):end);
            tIdx = strfind(imName, '-crop');
            imageNames = cat(1, imageNames, imName(1:(tIdx-1)));
        end
         
        subsaveDir = fullfile(saveDir, classNames{i}, j_instanceName); 
        if exist(fullfile(subsaveDir, 'doing.txt'), 'file')
            continue;
        end
        
        fid = fopen(fullfile(subsaveDir, 'doing.txt'), 'wt');
        fprintf(fid, 'doing...\n');
        fclose(fid);
        
        load(resFile);
        
        % top 1 classification accuracy        
        labelsIdxMat = cell2mat(labelsIdx); 
        labelsIdxMat = reshape(labelsIdxMat, 5, 1000); 
        labelsIdxMat = labelsIdxMat';
        labelsIdxMat1 = labelsIdxMat(:,1);
        
        ulabelsIdx = unique(labelsIdxMat1);
        predLabels = refLabels(ulabelsIdx);
        
        scoresMat = cell2mat(scores); 
        scoresMat = reshape(scoresMat, 5, 1000); 
        scoresMat = scoresMat';
        scoresMat1 = scoresMat(:,1);
        
        cnt = zeros(numel(ulabelsIdx),1); 
        meScores = zeros(numel(ulabelsIdx),1);
        assignments = zeros(nimgs, numel(ulabelsIdx));
        for k=1:numel(ulabelsIdx) 
            flag        = labelsIdxMat1 == ulabelsIdx(k);
            cnt(k)      = sum(flag); 
            meScores(k) = mean(scoresMat1(flag));
            assignments(flag, k) = 1;
        end
        
        [sortedCnt, sortIdx] =  sort(cnt, 'descend');
        sortPredLabels       =  predLabels(sortIdx);
        sortAssignments      =  assignments(:, sortIdx);
        sortMeScores         =  meScores(sortIdx);
        
        % keep only 90%
        cumCnt = cumsum(sortedCnt);
        for k=1:numel(ulabelsIdx)
            if cumCnt(k) > (0.9*nimgs)
                cutoffIdx = k;
                break;
            end
        end
        cutoffIdx           = min(cutoffIdx, numel(ulabelsIdx));
        c_sortedCnt         = sortedCnt(1:cutoffIdx);
        c_sortAssignments   = sortAssignments(:,1:cutoffIdx);
        c_sortMeScores      = sortMeScores(1:cutoffIdx);
        c_sortPredLabels    = sortPredLabels(1:cutoffIdx);
        

        figure;
        barweb(c_sortedCnt(:)', zeros(1, numel(c_sortedCnt)), 0.6, [], [], [], '#', [], ...
                            'y', c_sortPredLabels, 2, 'axis');
        set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
        title(['CNN: ' classNames{i} '-- '  j_instanceName], 'fontsize', 30);
        export_fig(fullfile(subsaveDir, 'statistics.png'), '-png', '-m1',   gcf);
        close all;
        
        % plot images
        nPredLabels = numel(c_sortPredLabels);
        nPlotRows = 6;
        nPlotCols = 9;
        for k=1:nPredLabels
            k_label = c_sortPredLabels{k};
            k_flag = logical( c_sortAssignments(:,k));
            k_imageFiles = imageFiles(k_flag);
            k_imageNames = imageNames(k_flag);
            k_n = sum(k_flag);
            
            nPlots = ceil(k_n/54);
            for m=1:nPlots
                s_showIdx = (m-1)*54+1;
                e_showIdx = min(m*54, k_n);
                
                figure;
                for n=s_showIdx:e_showIdx
                    subplot(nPlotCols, nPlotRows, n-s_showIdx+1);
                    imshow(imread(k_imageFiles{n}));
                    title(k_imageNames{n});
                end
                set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
                title(['gt-' classNames{i} '; pred-' k_label], 'fontsize', 20);
                export_fig(fullfile(subsaveDir, [classNames{i} '-- pred: ' k_label '-' num2str(m) '.png']), '-png', '-m1',   gcf);
                close all;
                
            end          
            
        end        

    end
end

% end