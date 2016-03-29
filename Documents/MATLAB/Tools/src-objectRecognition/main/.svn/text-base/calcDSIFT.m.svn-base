% calculate dense sift descriptors 


% load model
config;


% classify test images using CNN
saveDir = '/lab/ilab/30/jiaping/iLab20M/dense-sift';
load('testManifoldFileInfo.mat');

classNames = testFileInfo.classNames;
nClasses = numel(classNames);
classInfo = testFileInfo.classInfo;

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
            
        subsaveDir = fullfile(saveDir, classNames{i}, j_instanceName);
        if ~exist(subsaveDir, 'dir')
            mkdir(subsaveDir);
        else
            continue;
        end
        
        descriptors    =  cell(nimgs,1);
        fs =  cell(nimgs,1);
%         scores    =  cell(nimgs,1);
        fprintf(1, 'processing: %s-%s...\n', classNames{i}, j_instanceName);
        for k=1:nimgs
            
            im = imread(imageFiles{k});
%             [labels{k}, labelsIdx{k}, scores{k}] = cnnClassification(im, model);    
            [descriptors{k}, fs{k}] = calcDenseSIFT(im, 3, 2);
            
        end 
        save(fullfile(subsaveDir, 'siftDescriptors.mat'), 'descriptors', 'fs', '-v7.3');
    end
end
