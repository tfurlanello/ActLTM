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


% while 1
rng('shuffle');
rclassIdx = randperm(nClasses);

for ii=1:nClasses
    i = rclassIdx(ii);
    c_instanceInfo  = classInfo(i).instanceInfo;
    c_instanceNames = classInfo(i).instanceNames;
    nInstances      = numel(c_instanceNames);
    rinstanceIdx    = randperm(nInstances);
	subsaveDir = fullfile(saveDir, classNames{i}); 

    predLabelIdx    = [];
    predLabelScores = [];
    gtLabels = {};
    if exist(fullfile(subsaveDir, [classNames{i} '.png']), 'file')
        continue;
    end
    for jj=1:nInstances
        j = rinstanceIdx(jj);
        j_instanceName  = c_instanceInfo(j).instanceName;
        imageFiles      = c_instanceInfo(j).instanceFiles;
        nimgs           = numel(imageFiles);
        imageNames      = {};
        j_gtLabels = cell(nimgs,1);
        for k=1:nimgs
            j_gtLabels{k} = j_instanceName;
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
         
%         if exist(fullfile(subsaveDir, 'doing.txt'), 'file')
%             continue;
%         end
%         
%         fid = fopen(fullfile(subsaveDir, 'doing.txt'), 'wt');
%         fprintf(fid, 'doing...\n');
%         fclose(fid);
        
        load(resFile);
        
        % top 1 classification accuracy        
        labelsIdxMat = cell2mat(labelsIdx); 
        labelsIdxMat = reshape(labelsIdxMat, 5, 1000); 
        labelsIdxMat = labelsIdxMat';
        labelsIdxMat1 = labelsIdxMat(:,1);
        predLabelIdx = cat(1, predLabelIdx, labelsIdxMat1);
        gtLabels = cat(1, gtLabels, j_gtLabels);
                
        scoresMat = cell2mat(scores); 
        scoresMat = reshape(scoresMat, 5, 1000); 
        scoresMat = scoresMat';
        scoresMat1 = scoresMat(:,1);        
        predLabelScores = cat(1, predLabelScores, scoresMat1);

        ulabelsIdx = unique(labelsIdxMat1);
        predLabels = refLabels(ulabelsIdx);
        
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
        if cutoffIdx > 1
            titlestr = sprintf('CNN: %s--%s, %.3f-%.3f', classNames{i}, j_instanceName, ...
                            c_sortedCnt(1)/cumCnt(end), c_sortedCnt(2)/cumCnt(end));
        else
            titlestr = sprintf('CNN: %s--%s, %.3f', classNames{i}, j_instanceName, ...
                                    c_sortedCnt(1)/cumCnt(end));
        end
        title(titlestr, 'fontsize', 30);
        export_fig(fullfile(subsaveDir, [j_instanceName '.png']), '-png', '-m1',   gcf);
        close all;
        
        

    end
    
        tot_nimgs = numel(gtLabels);
        ulabelsIdx = unique(predLabelIdx);
        predLabels = refLabels(ulabelsIdx);

        
        cnt = zeros(numel(ulabelsIdx),1); 
        meScores = zeros(numel(ulabelsIdx),1);
        assignments = zeros(tot_nimgs, numel(ulabelsIdx));
        for k=1:numel(ulabelsIdx) 
            flag        = predLabelIdx == ulabelsIdx(k);
            cnt(k)      = sum(flag); 
            meScores(k) = mean(predLabelScores(flag));
            assignments(flag, k) = 1;
        end
        
        [sortedCnt, sortIdx] =  sort(cnt, 'descend');
        sortPredLabels       =  predLabels(sortIdx);
        sortAssignments      =  assignments(:, sortIdx);
        sortMeScores         =  meScores(sortIdx);
        
        % keep only 90%
        cumCnt = cumsum(sortedCnt);
        for k=1:numel(ulabelsIdx)
            if cumCnt(k) > (0.5*tot_nimgs)
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
        if cutoffIdx > 1
            titlestr = sprintf('CNN: %s, %.3f-%.3f', classNames{i}, ...
                            c_sortedCnt(1)/cumCnt(end), c_sortedCnt(2)/cumCnt(end));
        else
            titlestr = sprintf('CNN: %s--%s, %.3f', classNames{i}, ...
                                    c_sortedCnt(1)/cumCnt(end));
        end
        title(titlestr, 'fontsize', 30);
        export_fig(fullfile(subsaveDir, [classNames{i} '.png']), '-png', '-m1',   gcf);
        close all;
    
    
end
