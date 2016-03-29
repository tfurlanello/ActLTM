% generate training and testing images

saveDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/metadata-disentangling';

dataDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/metadata';

trainImgsInfoFile = 'trainImagesInfo.mat';
testImgsInfoFile  = 'testImagesInfo.mat';

classes = iLab_getClasses;
mapLabels = iLab_de_genLabelSpace;

trainFlag = fullfile(saveDir, 'train.flag');
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


    trainLabelsTransform = {};
    trainLabelsClass = {};
    trainImgFiles  = {};

    arg = iLab_validateImgFilePara;
    for i=1:nTrain

        if rem(i, 1000) == 0
            i
        end
        arg.class       =  classes{trainImagesInfo(i).objectIdx};
        arg.instance    =  trainImagesInfo(i).instanceIdx;
        arg.background  =  trainImagesInfo(i).backgroundIdx;
        arg.camera      =  trainImagesInfo(i).cameraIdx;
        arg.rotation    =  trainImagesInfo(i).rotationIdx;
        arg.light       =  trainImagesInfo(i).lightIdx;
        arg.focus       =  trainImagesInfo(i).focusIdx;

        ref_img_file = iLab_genImgFileName(arg);
        neighbors = iLab_de_getImgNeighbors(arg);

        for n=1:numel(neighbors)
           neiIdx = [neighbors{n}.background iLab_getClassIdx(neighbors{n}.class) ...
               neighbors{n}.instance, neighbors{n}.camera, neighbors{n}.rotation, ...
               neighbors{n}.focus, neighbors{n}.light];
           neiIdx = neiIdx(:);

           if any( sum(bsxfun(@eq, values, neiIdx),1) == 7)
               [l_class, l_transform] = ...
                                iLab_de_genLabels4imgPairs(arg, neighbors{n});
                trainLabelsClass     = cat(1, trainLabelsClass, l_class);
                trainLabelsTransform = cat(1, trainLabelsTransform, l_transform);
                tar_img_file    =  iLab_genImgFileName(neighbors{n});
                trainImgFiles   =  cat(1, trainImgFiles, {ref_img_file tar_img_file});

           else
               continue;
           end
        end

    end
    
    
    save(fullfile(saveDir, 'train.mat'), 'trainLabelsClass', 'trainLabelsTransform', ...
                    'trainImgFiles');

end


testFlag = fullfile(saveDir, 'test.flag');

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

    testLabelsTransform = {};
    testLabelsClass = {};
    testImgFiles  = {};

    arg = iLab_validateImgFilePara;
    for i=1:nTest

        if rem(i, 1000) == 0
            i
        end

        arg.class       =  classes{testImagesInfo(i).objectIdx};
        arg.instance    =  testImagesInfo(i).instanceIdx;
        arg.background  =  testImagesInfo(i).backgroundIdx;
        arg.camera      =  testImagesInfo(i).cameraIdx;
        arg.rotation    =  testImagesInfo(i).rotationIdx;
        arg.light       =  testImagesInfo(i).lightIdx;
        arg.focus       =  testImagesInfo(i).focusIdx;

        ref_img_file = iLab_genImgFileName(arg);
        neighbors = iLab_de_getImgNeighbors(arg);

        for n=1:numel(neighbors)
           neiIdx = [neighbors{n}.background iLab_getClassIdx(neighbors{n}.class) ...
               neighbors{n}.instance, neighbors{n}.camera, neighbors{n}.rotation, ...
               neighbors{n}.focus, neighbors{n}.light];
           neiIdx = neiIdx(:);

           if any( sum(bsxfun(@eq, values, neiIdx),1) == 7)
               [l_class, l_transform] = ...
                                iLab_de_genLabels4imgPairs(arg, neighbors{n});
                testLabelsClass = cat(1, testLabelsClass, l_class);
                testLabelsTransform = cat(1, testLabelsTransform, l_transform);
                tar_img_file = iLab_genImgFileName(neighbors{n});
                testImgFiles = cat(1, testImgFiles, {ref_img_file tar_img_file});

           else
               continue;
           end
        end

    end
    
    
    save(fullfile(saveDir, 'test.mat'), 'testLabelsClass',  'testLabelsTransform', ...
                    'testImgFiles');    
end






