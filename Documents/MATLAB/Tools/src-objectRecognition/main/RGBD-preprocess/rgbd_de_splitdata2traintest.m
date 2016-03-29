% this script is used to split data into training and testing 
% we use leave-one-out as the mean to split data into training and testing

saveDir = '/lab/igpu3/u/jiaping/washington-RGBD/dataset/experiments';

% images with object center and scale information
imInfoFile = '/lab/jiaping/svn-jiaping/projects/iLab-object-recognition/src/main/RGBD-data-info/imgInfo-center-scale.mat';
load(imInfoFile, 'imInfo');
imHierarchy = rgbd_getImgHierarchy('raw');   

cameras     = rgbd_getCameraIdx;
nCameras    = numel(cameras);
categories  = rgbd_getClasses;
nCategories = numel(categories);

nImages       = numel(imInfo);


%% first generate training and testing lists, and then within each 
%% list, generate image pairs, which will be trained by disentangling
%% CNN framework

% prepare 4 different leave-one-out splits
% within each split, prepare image pairs, but with different # of
% camera-pair labels
nExperiments = 4;
repetitions = [1 2 3 4 5];
nRep = numel(repetitions);
gap = 5;

for e=1:nExperiments
    
    subsaveDir = fullfile(saveDir, ['exp-' num2str(e)]);
    if ~exist(subsaveDir, 'dir')
        mkdir(subsaveDir);
    end     

    im_categories = {imInfo.class};
    im_instances  = cell2mat({imInfo.instance});
    im_cameras    = cell2mat({imInfo.camera});
    im_frames     = cell2mat({imInfo.frame});
    
    if exist(fullfile(subsaveDir, 'split.mat'), 'file')
        continue;
    end
    
    % (1) split
    bTrain = zeros(1,nImages) > 1.0;
    bTest  = zeros(1,nImages) > 1.0;
    for c=1:nCategories
        c_classes = categories{c};
        bClass = strcmp(c_classes, im_categories);    
        instances   =  imHierarchy.(c_classes).instancesID;
        nInstances  =  numel(instances);    
        testInstance    = instances(randi(nInstances));
    %     trainInstances  = setdiff(instances,testInstance);
        bInstance = testInstance == im_instances;    
        bTest = bTest | (bClass & bInstance);        
    end
    bTrain  = ~bTest;
    save(fullfile(subsaveDir, 'split.mat'), 'bTrain', 'bTest', 'imInfo');
    
    for rep=1:nRep
        fprintf(1, 'processing: exp-%d/%d, rep-%d/%d\n', e, nExperiments, rep, nRep);

        % (2) prepare image-pairs with camera-pair labels
        repetition = repetitions(rep);
        %% training
        imInfo_train = imInfo(bTrain);
        im_categories = {imInfo_train.class};
        im_instances  = cell2mat({imInfo_train.instance});
        im_cameras    = cell2mat({imInfo_train.camera});
        im_frames     = cell2mat({imInfo_train.frame});

        if exist(fullfile(subsaveDir, ['train' '-rep' num2str(rep) '.txt']), 'file')
            continue;
        end
        
        fid = fopen(fullfile(subsaveDir, ['train' '-rep' num2str(rep) '.txt']), 'w');
        for c=1:nCategories
           c_classes = categories{c};
           bClass = strcmp(c_classes, im_categories);

           instances = imHierarchy.(c_classes).instancesID;
           nInstances = numel(instances);

           for i=1:nInstances
              bInstance = instances(i) == im_instances;

              for cam=1:nCameras
                 c_camera = cameras(cam);
                 bCamera  = c_camera == im_cameras;

                 bflag = bClass & bInstance & bCamera;
                 if sum(bflag) == 0
                     continue;
                 end

                 % prepare image pairs , pair labels
                 tar_imInfo = imInfo_train(bflag);
                 tar_frames = cell2mat({tar_imInfo.frame});
                 tar_cameras = cell2mat({tar_imInfo.camera});
                 cameraIdx = unique(tar_cameras);
                 assert(isscalar(cameraIdx));

                [pairs_frame, pairs_label, map2frameIdx] = ...
                    rgbd_genImgPairs(tar_frames, cameraIdx, gap, repetition);

                nPairs = size(pairs_frame,1);

                for n=1:nPairs
                   im_l = rgbd_genImgFileName(tar_imInfo(map2frameIdx(n,1)), 'raw');
                   im_r = rgbd_genImgFileName(tar_imInfo(map2frameIdx(n,2)), 'raw');
                   im_l = [im_l(1:end-4) '.jpg'];
                   im_r = [im_r(1:end-4) '.jpg'];

                   t_label = rgbd_de_genLabels(pairs_label(n,1), pairs_label(n,2));

                   fprintf(fid, '%s %s %s %s\n', c_classes, t_label, im_l, im_r);


                end
              end
           end
        end
        fclose(fid);

        %% testing
        imInfo_test = imInfo(bTest);
        im_categories = {imInfo_test.class};
        im_instances  = cell2mat({imInfo_test.instance});
        im_cameras    = cell2mat({imInfo_test.camera});
        im_frames     = cell2mat({imInfo_test.frame});

        if exist(fullfile(subsaveDir, ['test' '-rep' num2str(rep) '.txt']), 'file')
            continue;
        end
        
        fid = fopen(fullfile(subsaveDir, ['test' '-rep' num2str(rep) '.txt']), 'w');
        for c=1:nCategories
           c_classes = categories{c};
           bClass = strcmp(c_classes, im_categories);

           instances = imHierarchy.(c_classes).instancesID;
           nInstances = numel(instances);

           for i=1:nInstances
              bInstance = instances(i) == im_instances;

              for cam=1:nCameras
                 c_camera = cameras(cam);
                 bCamera  = c_camera == im_cameras;

                 bflag = bClass & bInstance & bCamera;
                 if sum(bflag) == 0
                     continue;
                 end

                 % prepare image pairs , pair labels
                 tar_imInfo = imInfo_test(bflag);
                 tar_frames = cell2mat({tar_imInfo.frame});
                 tar_cameras = cell2mat({tar_imInfo.camera});
                 cameraIdx = unique(tar_cameras);
                 assert(isscalar(cameraIdx));

                [pairs_frame, pairs_label, map2frameIdx] = ...
                    rgbd_genImgPairs(tar_frames, cameraIdx, gap, repetition);

                nPairs = size(pairs_frame,1);

                for n=1:nPairs
                   im_l = rgbd_genImgFileName(tar_imInfo(map2frameIdx(n,1)), 'raw');
                   im_r = rgbd_genImgFileName(tar_imInfo(map2frameIdx(n,2)), 'raw');
                   im_l = [im_l(1:end-4) '.jpg'];
                   im_r = [im_r(1:end-4) '.jpg'];

                   t_label = rgbd_de_genLabels(pairs_label(n,1), pairs_label(n,2));

                   fprintf(fid, '%s %s %s %s\n', c_classes, t_label, im_l, im_r);


                end
              end
           end
        end
        fclose(fid);
    
    end

 

end