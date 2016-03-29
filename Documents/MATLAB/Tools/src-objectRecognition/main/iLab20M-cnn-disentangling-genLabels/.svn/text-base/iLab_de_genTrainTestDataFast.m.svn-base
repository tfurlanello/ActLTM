% generate training and testing images

saveDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/metadata-disentangling';
dataDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/metadata';

trainImgsInfoFile  =  'trainImagesInfo.mat';
testImgsInfoFile   =  'testImagesInfo.mat';

classes = iLab_getClasses;
mapLabels = iLab_de_genLabelSpace;

mapCameraNei = iLab_de_getNeighboringCameras;
mapCameraNei = containers.Map(mapCameraNei(:,1), mapCameraNei(:,2));

separator = '-';
trainFlag = fullfile(saveDir, 'train-fast.flag');
if ~exist(trainFlag, 'file')
    fid = fopen(trainFlag, 'w');
    fprintf(fid, '1\n');
    fclose(fid);
 
    %% training images
    load(fullfile(dataDir, trainImgsInfoFile));
    nTrain = numel(trainImagesInfo);
    names = fieldnames(trainImagesInfo);
    values = [];
    for f=1:numel(names)
        values = cat(1, values, cell2mat({trainImagesInfo.(names{f})}));
    end

    
    trainClasses = unique(classes(values(2,:)));
    mapClasses = containers.Map(trainClasses, 1:numel(trainClasses));
    
   
    bNeighbors1 = zeros(nTrain,1) < -1.0;
    bNeighbors2 = zeros(nTrain,1) < -1.0;
    
    valNeighbors1 = values;
    valNeighbors2 = values;
    
    parfor i=1:nTrain
        
        if rem(i, 1000) == 0
            i
        end
        
        ref_para     =  values(:,i);
        ref_camera   =  ref_para(4);
        ref_rotation =  ref_para(5);
                  
        % neighboring camera
        if mapCameraNei.isKey(ref_camera)
            
            bNeighbors1(i) = true;
            tar_camera = mapCameraNei(ref_camera);  
            tar_para = ref_para;
            tar_para(4) = tar_camera;
            valNeighbors1(:,i) = tar_para;

%             tar_para = ref_para;
%             tar_camera = mapCameraNei(ref_camera);             
%             tar_para(4) = tar_camera;
%             
%              if any( sum(bsxfun(@eq, values, tar_para),1) == 7)
%                 bNeighbors1(i) = true;
%                 tar_camera = mapCameraNei(ref_camera);
%                 valNeighbors1(4,i) = tar_camera;
%              end
        end
        
        % neighboring rotation
        tar_rotation = ref_rotation + 1;
        if tar_rotation < 8   
            bNeighbors2(i) = true;
            tar_para = ref_para;
            tar_para(5) = tar_rotation;
            valNeighbors2(:,i) = tar_para;
            
%             tar_para = ref_para;
%             tar_para(5) = tar_rotation;  
%              if any( sum(bsxfun(@eq, values, tar_para),1) == 7)
%                 valNeighbors2(5,i) = tar_rotation;
%                 bNeighbors2(i) = true;
%              end            
        end
        
    end
    
    trainLabelsTransform1 = cell(nTrain,1);
    trainLabelsClass1 = cell(nTrain,1);
    trainImgFiles1  = cell(nTrain,1);
    
	trainLabelsTransform2 = cell(nTrain,1);
    trainLabelsClass2 = cell(nTrain,1);
    trainImgFiles2  = cell(nTrain,1);
    
    parfor i=1:nTrain
        
        if rem(i, 1000) == 0
            i            
        end
        
        ref_para    =   values(:,i);
        tar_para1   =   valNeighbors1(:,i); % camera
        tar_para2   =   valNeighbors2(:,i); % rotation
        b_tar1      =   bNeighbors1(i);
        b_tar2      =   bNeighbors2(i);
        
        class = classes{ref_para(2)};
        ref_img_file = iLab_genImgFileName({'class', class, 'instance', ref_para(3), ...
            'background', ref_para(1), 'camera', ref_para(4), 'rotation', ref_para(5), ...
            'focus', ref_para(6), 'light', ref_para(7)});
        

        if b_tar1
            % neigboring camera
            tar_img_file1 = iLab_genImgFileName({'class', class, 'instance', tar_para1(3), ...
                'background', tar_para1(1), 'camera', tar_para1(4), 'rotation', tar_para1(5), ...
                'focus', tar_para1(6), 'light', tar_para1(7)});     

            l_c1 = iLab_idx2nameCamera(ref_para(4));
            l_c2 = iLab_idx2nameCamera(tar_para1(4));
            l_r  = iLab_idx2nameRotation(ref_para(5));
            
            transform = [l_c1 separator l_c2 separator l_r]; 
            
            trainLabelsClass1{i} = class;
            trainLabelsTransform1{i} = transform;
            trainImgFiles1{i} = {ref_img_file tar_img_file1};
            
        end

        if b_tar2
            % neighboring rotation
            tar_img_file2 = iLab_genImgFileName({'class', class, 'instance', tar_para2(3), ...
                'background', tar_para2(1), 'camera', tar_para2(4), 'rotation', tar_para2(5), ...
                'focus', tar_para2(6), 'light', tar_para2(7)});
            
            l_c  = iLab_idx2nameCamera(ref_para(4));    
            l_r1  = iLab_idx2nameRotation(ref_para(5));
            l_r2  = iLab_idx2nameRotation(tar_para2(5));
            
            transform = [l_c  separator l_r1 separator l_r2];  
            
            trainLabelsClass2{i} = class;
            trainLabelsTransform2{i} = transform;
            trainImgFiles2{i} = {ref_img_file tar_img_file2};
            
        end
        
    end
    
    flag1 = bNeighbors1;
    flag2 = bNeighbors2;
    
    save(fullfile(saveDir, 'train-fast.mat'), 'trainLabelsClass1', 'trainLabelsTransform1', ...
                    'trainImgFiles1',  'trainLabelsClass2', 'trainLabelsTransform2', ...
                    'trainImgFiles2', 'flag1', 'flag2', 'mapLabels', 'mapClasses');

end


testFlag = fullfile(saveDir, 'test-fast.flag');

if ~exist(testFlag, 'file')
    fid = fopen(testFlag, 'w');
    fprintf(fid, '1\n');
    fclose(fid);

    %% test images
    load(fullfile(dataDir, testImgsInfoFile));
    nTest = numel(testImagesInfo);
    names = fieldnames(testImagesInfo);
    values = [];
    for f=1:numel(names)
        values = cat(1, values, cell2mat({testImagesInfo.(names{f})}));
    end
 

    bNeighbors1 = zeros(nTest,1) < -1.0;
    bNeighbors2 = zeros(nTest,1) < -1.0;
    
    valNeighbors1 = values;
    valNeighbors2 = values;
    
    parfor i=1:nTest
        
        if rem(i, 1000) == 0
            i
        end
        
        ref_para     =  values(:,i);
        ref_camera   =  ref_para(4);
        ref_rotation =  ref_para(5);
                  
        % neighboring camera
        if mapCameraNei.isKey(ref_camera)
            
            bNeighbors1(i) = true;
            tar_camera = mapCameraNei(ref_camera);  
            tar_para = ref_para;
            tar_para(4) = tar_camera;
            valNeighbors1(:,i) = tar_para;

        end
        
        % neighboring rotation
        tar_rotation = ref_rotation + 1;
        if tar_rotation < 8   
            bNeighbors2(i) = true;
            tar_para = ref_para;
            tar_para(5) = tar_rotation;
            valNeighbors2(:,i) = tar_para;
         
        end
        
    end
    
    testLabelsTransform1 = cell(nTest,1);
    testLabelsClass1 = cell(nTest,1);
    testImgFiles1  = cell(nTest,1);
    
	testLabelsTransform2 = cell(nTest,1);
    testLabelsClass2 = cell(nTest,1);
    testImgFiles2  = cell(nTest,1);
    
    parfor i=1:nTest
        
        if rem(i, 1000) == 0
            i            
        end
        
        ref_para    =   values(:,i);
        tar_para1   =   valNeighbors1(:,i); % camera
        tar_para2   =   valNeighbors2(:,i); % rotation
        b_tar1      =   bNeighbors1(i);
        b_tar2      =   bNeighbors2(i);
        
        class = classes{ref_para(2)};
        ref_img_file = iLab_genImgFileName({'class', class, 'instance', ref_para(3), ...
            'background', ref_para(1), 'camera', ref_para(4), 'rotation', ref_para(5), ...
            'focus', ref_para(6), 'light', ref_para(7)});
        

        if b_tar1
            % neigboring camera
            tar_img_file1 = iLab_genImgFileName({'class', class, 'instance', tar_para1(3), ...
                'background', tar_para1(1), 'camera', tar_para1(4), 'rotation', tar_para1(5), ...
                'focus', tar_para1(6), 'light', tar_para1(7)});     

            l_c1 = iLab_idx2nameCamera(ref_para(4));
            l_c2 = iLab_idx2nameCamera(tar_para1(4));
            l_r  = iLab_idx2nameRotation(ref_para(5));
            
            transform = [l_c1 separator l_c2 separator l_r]; 
            
            testLabelsClass1{i} = class;
            testLabelsTransform1{i} = transform;
            testImgFiles1{i} = {ref_img_file tar_img_file1};
            
        end

        if b_tar2
            % neighboring rotation
            tar_img_file2 = iLab_genImgFileName({'class', class, 'instance', tar_para2(3), ...
                'background', tar_para2(1), 'camera', tar_para2(4), 'rotation', tar_para2(5), ...
                'focus', tar_para2(6), 'light', tar_para2(7)});
            
            l_c  = iLab_idx2nameCamera(ref_para(4));    
            l_r1  = iLab_idx2nameRotation(ref_para(5));
            l_r2  = iLab_idx2nameRotation(tar_para2(5));
            
            transform = [l_c  separator l_r1 separator l_r2];  
            
            testLabelsClass2{i} = class;
            testLabelsTransform2{i} = transform;
            testImgFiles2{i} = {ref_img_file tar_img_file2};
            
        end
        
    end
    
    flag1 = bNeighbors1;
    flag2 = bNeighbors2;
    
    save(fullfile(saveDir, 'test-fast.mat'), 'testLabelsClass1', 'testLabelsTransform1', ...
                    'testImgFiles1',  'testLabelsClass2', 'testLabelsTransform2', ...
                    'testImgFiles2', 'flag1', 'flag2');
end




if exist(fullfile(saveDir, 'train-fast.mat'), 'file')
    
    load(fullfile(saveDir, 'train-fast.mat'));
    nTrain = numel(flag1);
    
    fid = fopen(fullfile(saveDir, 'train.txt'), 'w');
    
    for i=1:nTrain
        
        if rem(i,1000) == 0
            i
        end
        if flag1(i) == true
            class = trainLabelsClass1{i};
            transform = trainLabelsTransform1{i};
            imgPairs = trainImgFiles1{i};
            
            classIdx = mapClasses(class);
            transformIdx = mapLabels(transform);
            
            fprintf(fid, '%s %d %s %d %s %s\n', class, classIdx, ...
                            transform, transformIdx, imgPairs{1}, imgPairs{2});
            
        end
        
        if flag2(i) == true
            class = trainLabelsClass2{i};
            transform = trainLabelsTransform2{i};
            imgPairs = trainImgFiles2{i};
            
            classIdx = mapClasses(class);
            transformIdx = mapLabels(transform);
            
            fprintf(fid, '%s %d %s %d %s %s\n', class, classIdx, ...
                            transform, transformIdx, imgPairs{1}, imgPairs{2});
            
            
        end
        
    end
    
    fclose(fid);
    
end



if exist(fullfile(saveDir, 'test-fast.mat'), 'file')
    
    load(fullfile(saveDir, 'test-fast.mat'));
    nTrain = numel(flag1);
    
    fid = fopen(fullfile(saveDir, 'test.txt'), 'w');
    
    for i=1:nTrain
        
        if rem(i,1000) == 0
            i
        end
        if flag1(i) == true
            class   = testLabelsClass1{i};
            transform = testLabelsTransform1{i};
            imgPairs = testImgFiles1{i};
            
            classIdx = mapClasses(class);
            transformIdx = mapLabels(transform);
            
            fprintf(fid, '%s %d %s %d %s %s\n', class, classIdx, ...
                            transform, transformIdx, imgPairs{1}, imgPairs{2});
            
        end
        
        if flag2(i) == true
            class = testLabelsClass2{i};
            transform = testLabelsTransform2{i};
            imgPairs = testImgFiles2{i};
            
            classIdx = mapClasses(class);
            transformIdx = mapLabels(transform);
            
            fprintf(fid, '%s %d %s %d %s %s\n', class, classIdx, ...
                            transform, transformIdx, imgPairs{1}, imgPairs{2});
            
            
        end
        
    end
    
    fclose(fid);
    
end




