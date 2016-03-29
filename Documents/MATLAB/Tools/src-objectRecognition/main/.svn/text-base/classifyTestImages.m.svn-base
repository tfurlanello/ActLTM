% load model
config;
global cnnModel;
model = load(cnnModel) ;


% classify test images using CNN
saveDir = '/lab/ilab/30/jiaping/iLab20M';
load('testFileInfo.mat');

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
        
        labels    =  cell(nimgs,1);
        labelsIdx =  cell(nimgs,1);
        scores    =  cell(nimgs,1);
        fprintf(1, 'processing: %s-%s...\n', classNames{i}, j_instanceName);
        parfor k=1:nimgs
            
            im = imread(imageFiles{k});
            [labels{k}, labelsIdx{k}, scores{k}] = cnnClassification(im, model);    
            
        end 
        save(fullfile(subsaveDir, 'predictedLabels.mat'), 'labels', 'labelsIdx', 'scores');
    end
end

