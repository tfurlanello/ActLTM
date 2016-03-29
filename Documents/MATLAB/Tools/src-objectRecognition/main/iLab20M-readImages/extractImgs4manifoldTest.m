% 1. extract images under lighting-0-focus-0
% 2. extract dense sift from each image
% 3. use bag-of-words to encode each images
% 4. use t-SNE to visualize images


dirTree     = iLab_getDataDirTree;
classNames  = dirTree.classNames;
nClasses    = numel(classNames);

strPattern = '\W*l0-f0\W*';
nInst = 5;
nBackG = 5;
classes = {'car', 'mil', 'tank', 'boat'};

classInfo = struct('instanceNames', {}, 'instanceInfo', {});
cnt = 0;
classNames = {};
for i=1:nClasses
    i
    idx = strfind(classes, dirTree.classNames{i});
    if sum(cell2mat(idx)) == 0
        continue;
    end
    classNames = cat(1, classNames, dirTree.classNames{i});
    c_instanceNames = dirTree.classInfo(i).instanceNames;
    nInstances = numel(c_instanceNames);
    instanceInfo = struct('instanceName', {}, 'instanceFiles', {});
    for j=1:min(nInstances, nInst)
        j_instanceDir       =   dirTree.classInfo(i).instanceInfo(j).instanceDir;
        j_instanceBGs       =   dirTree.classInfo(i).instanceInfo(j).backgroundNames;
        j_instanceName      =   dirTree.classInfo(i).instanceNames{j};
        nBG                 =   numel(j_instanceBGs);
        j_instanceFileNames = {};
        
        sBG = max(1,nBG - nBackG + 1);
        for k=sBG:nBG
            bg_dir = fullfile(j_instanceDir, j_instanceBGs{k});
            tmp = getImgFiles(bg_dir, '.png');
            
            idx = regexp(tmp, strPattern);
            flag = zeros(numel(idx),1) > 1.0;
            for m=1:numel(idx)
                flag(m) = (~isempty(idx{m}));
            end
            j_instanceFileNames = cat(1,j_instanceFileNames, fullfile(bg_dir, tmp(flag)));            
        end

        imageFiles = j_instanceFileNames;
        instanceInfo(j).instanceName = j_instanceName;
        instanceInfo(j).instanceFiles = imageFiles;
    end
    cnt = cnt + 1;
    classInfo(cnt).instanceNames = c_instanceNames(1:min(nInstances, nInst));
    classInfo(cnt).instanceInfo = instanceInfo;
%     save(fullfile(savedir, ['classinfo' num2str(i) '.mat']), 'classInfo');
end


testFileInfo.classNames = classNames;
testFileInfo.classInfo = classInfo;
save('testManifoldFileInfo.mat', 'testFileInfo');



