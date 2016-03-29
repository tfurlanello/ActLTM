%% use bag-of-words representation to describe each image


% (1) add clustering path
addpath(genpath('/lab/jiaping/projects/google-glass-project/src/clustering'));


% (2) load dense SIFT descriptors
load('/lab/ilab/30/jiaping/iLab20M/dense-sift/car/car-0004/siftDescriptors.mat');


saveDir = '/lab/ilab/30/jiaping/iLab20M/dense-sift';
load('testManifoldFileInfo.mat');

classNames = testFileInfo.classNames;
nClasses = numel(classNames);
classInfo = testFileInfo.classInfo;

rng('shuffle');
rclassIdx = randperm(nClasses);

nclusters = [50 100 200 300];
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
            
        subsaveDir = fullfile(saveDir, classNames{i}, j_instanceName);
        if ~exist(fullfile(subsaveDir, 'siftDescriptors.mat'), 'file')            
            continue;
        end
        
        if exist(fullfile(subsaveDir, 'bow.txt'), 'file')
            continue;
        end
        
        
        fid = fopen(fullfile(subsaveDir, 'bow.txt'), 'wt');
        fprintf(fid, '1\n');
        fclose(fid);
        
        
        load(fullfile(subsaveDir, 'siftDescriptors.mat'));
        
        % (3) do kmeans clustering
        nImgs = numel(descriptors);

%         X = [];
        cnt = zeros(nImgs,1);
        for k=1:nImgs
            [k, nImgs]
%             X   = cat(2,X, descriptors{k});
            cnt(k) = size(descriptors{k},2);
        end
        cuCnt = [0; cumsum(cnt)];
        
        X = cell2mat(descriptors');
        clear descriptors fs;

        X = X';  
        size(X)
        clust_method = 'kmeans';
        param = validateKMeansparam;
        bow = cell(nImgs,1);
 
        for nclust=1:numel(nclusters)
            fprintf(1, 'cluster #:%d\n', nclusters(nclust));
            ncluster = nclusters(nclust);
            param.nclusters = ncluster;
            partitions = doClustering(X, clust_method, param);

            for k=1:nImgs
                k_assignments = partitions((cuCnt(k)+1):cuCnt(k+1));
                tmp_cnt = zeros(ncluster,1);
                for m=1:ncluster
                    tmp_cnt(m) = sum( k_assignments == m);
                end
                bow{k} = tmp_cnt;
            end

            save(fullfile(subsaveDir, ['bow' num2str(ncluster) '.mat']), 'bow');

        end
        clear X;

    end
end