
% randomly choose some images from each instance, and do classification

dirTree     = iLab_getDataDirTreeiLab;
classNames  = dirTree.classNames;
nClasses    = numel(classNames);

nImg = 1000;
classInfo = struct('instanceNames', {}, 'instanceInfo', {});

% rng('shuffle');
% ridx = randperm(nClasses);
% savedir = './testData';
for i=1:nClasses
    i
%     i = ridx(ii);
%     if exist(fullfile(savedir, [num2str(i) '.txt']), 'file')
%         continue;
%     end
%     i
%     fid = fopen(fullfile(savedir, [num2str(i) '.txt']), 'wt');
%     fprintf(fid, '1\n');
%     fclose(fid);
    c_instanceNames = dirTree.classInfo(i).instanceNames;
    nInstances = numel(c_instanceNames);
    instanceInfo = struct('instanceName', {}, 'instanceFiles', {});
    for j=1:nInstances
        j_instanceDir = dirTree.classInfo(i).instanceInfo(j).instanceDir;
        j_instanceBGs = dirTree.classInfo(i).instanceInfo(j).backgroundNames;
        j_instanceName = dirTree.classInfo(i).instanceNames{j};
        nBG = numel(j_instanceBGs);
        j_instanceFileNames = {};
        for k=1:nBG
            bg_dir = fullfile(j_instanceDir, j_instanceBGs{k});
            tmp = getImgFiles(bg_dir, '.png');
            j_instanceFileNames = cat(1,j_instanceFileNames, fullfile(bg_dir, tmp));            
        end
        rng('shuffle');
        ridx = randperm(numel(j_instanceFileNames));
        imageFiles = j_instanceFileNames(ridx(1:min(nImg,numel(j_instanceFileNames))));
        instanceInfo(j).instanceName = j_instanceName;
        instanceInfo(j).instanceFiles = imageFiles;
    end
    
    classInfo(i).instanceNames = c_instanceNames;
    classInfo(i).instanceInfo = instanceInfo;
%     save(fullfile(savedir, ['classinfo' num2str(i) '.mat']), 'classInfo');
end


testFileInfo.classNames = classNames;
testFileInfo.classInfo = classInfo;

save('testFileInfo.mat', 'testFileInfo');





