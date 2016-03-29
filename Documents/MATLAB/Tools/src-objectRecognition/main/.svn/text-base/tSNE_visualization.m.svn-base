
% visualization of BOW

% (1) add clustering path
addpath(genpath('/lab/jiaping/projects/google-glass-project/src/clustering'));

saveDir = '/lab/ilab/30/jiaping/iLab20M/dense-sift';
load('testManifoldFileInfo.mat');

classNames = testFileInfo.classNames;
nClasses = numel(classNames);
classInfo = testFileInfo.classInfo;

rng('shuffle');
rclassIdx = randperm(nClasses);

cameraStr = {'c00', 'c01', 'c02', 'c03', 'c04', ...
             'c05', 'c06', 'c07', 'c08', 'c09', 'c10'};
nCameras = numel(cameraStr);
nBackgrounds = 5;
nImgBackGround = 88;


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
         
        imageNames = {};
        for k=1:nimgs
            slashIdx = strfind(imageFiles{k}, '/');
            imName = imageFiles{k}((slashIdx(end)+1):end);
             imageNames = cat(1, imageNames, imName );
        end
        
        
        subsaveDir = fullfile(saveDir, classNames{i}, j_instanceName);
  
        
        for k=1:numel(nclusters)
            ncluster = nclusters(k);
            if ~exist(fullfile(subsaveDir, ['bow' num2str(ncluster) '.mat']), 'file')
                continue;
            end
            
            load(fullfile(subsaveDir, ['bow' num2str(ncluster) '.mat']));
            
%             for m=1:nBackgrounds
%                 sBG = (m-1)*nImgBackGround + 1;
%                 eBG = m*nImgBackGround;
%                 
%                 m_imageNames = imageNames(sBG:eBG);
%                 m_labels = reshape(repmat(1:11, 8,1), 88,1);
%                 m_bow = bow(sBG:eBG);
%             end
%             
      
            m_bow = cell2mat(bow);
            m_bow = reshape(m_bow, round(numel(m_bow(:))/numel(bow)), numel(bow));
            m_bow = m_bow';
%             labels = zeros(nimgs,1);
            
%             for c=1:nCameras
%                 
%                 idx = regexp(imageNames, cameraStr{c});
%                 flag = zeros(numel(idx),1) > 1.0;
%                 for m=1:numel(idx)
%                     flag(m) = (~isempty(idx{m}));
%                 end                
%                 labels(flag) = c;     
%             end

            labels = reshape(repmat(1:5, 88,1), 440,1);            
            [mapped2D, landmarks] = visualizeDescriptorsTSNE(m_bow, labels);            
            imageFiles1 = imageFiles(landmarks);
            
            save(fullfile(subsaveDir, sprintf('mapped2D-cluster-%d.mat', ncluster)), ...
                'imageFiles', 'mapped2D');
            export_fig(fullfile(subsaveDir, sprintf('mapped2D-cluster-%d.png', ncluster)), ...
                '-png', '-m1',   gcf);
            
            [nonsquare, square] = tSNEcollage(mapped2D, imageFiles1);
            imwrite(nonsquare, fullfile(subsaveDir, sprintf('collage-s-cluster-%d.png', ncluster)));
            imwrite(square, fullfile(subsaveDir, sprintf('collage-ns-cluster-%d.png', ncluster)) );
            
            close all;
            
        end
        
    

    end
end