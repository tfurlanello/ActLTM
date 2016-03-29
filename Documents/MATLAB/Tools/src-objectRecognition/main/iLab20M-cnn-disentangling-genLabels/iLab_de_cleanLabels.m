function iLab_de_cleanLabels(dataDir)

    % make sure: images in the training and testing sets exist!
    trainImgDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/trainImages';
    testImgDir  = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/testImages';
%     dataDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/metadata-disentangling-rep5';


    %% read training images
    tic
    file = dir(strcat(trainImgDir, filesep, '*.jpg'));
    trainImgNames = {file.name};
    toc

    load(fullfile(dataDir, 'mappings-f56.mat'));
    
    %% original training lists
%     [a,b,c,d,e,f] = textread(fullfile(dataDir, 'train-candidate.txt'), '%s %s %s %s %s %s\n');
    [a,b,c,d,e,f] = textread(fullfile(dataDir, 'train-f56.txt'), '%s %s %s %s %s %s\n');
    
%     d = mapTransform(c);
    
    pairL = e;
    pairR = f;
    bTrain = zeros(numel(pairR),1) < 1.0;
    nTrain = numel(pairR);

    assert(isempty(setdiff(pairL, trainImgNames)));

    noexistImgs = setdiff(pairR, trainImgNames); 
    for i=1:numel(noexistImgs)      
        idx = find(strcmp(pairR, noexistImgs{i}));
        bTrain(idx) = false;        
    end

    fid = fopen(fullfile(dataDir, 'train.txt'), 'w');
    for i=1:nTrain

        if rem(i,10000) == 0
            i
        end

        if bTrain(i)
            fprintf(fid, '%s %s %s %d %s %s\n', a{i}, b{i}, c{i}, mapTransform(c{i}), e{i}, f{i});
%             fprintf(fid, '%s %s %s %s %s %s\n', a{i}, b{i}, c{i}, d{i}, e{i}, f{i});
        end

    end
    fclose(fid);



    %% read test images
    tic
    file = dir(strcat(testImgDir, filesep, '*.jpg'));
    testImgNames = {file.name};
    toc

    %% original training lists
%     [a,b,c,d,e,f] = textread(fullfile(dataDir, 'test-candidate.txt'), '%s %s %s %s %s %s\n');
    [a,b,c,d,e,f] = textread(fullfile(dataDir, 'test-f56.txt'), '%s %s %s %s %s %s\n');

%     d = mapTransform(c);
    pairL = e;
    pairR = f;
    bTest = zeros(numel(pairR),1) < 1.0;
    nTest = numel(pairR);

    assert(isempty(setdiff(pairL, testImgNames)));

    noexistImgs = setdiff(pairR, testImgNames); 
    for i=1:numel(noexistImgs)      
        idx = find(strcmp(pairR, noexistImgs{i}));
        bTest(idx) = false;        
    end

    fid = fopen(fullfile(dataDir, 'test.txt'), 'w');
    for i=1:nTest

        if rem(i,10000) == 0
            i
        end

        if bTest(i)
            fprintf(fid, '%s %s %s %d %s %s\n', a{i}, b{i}, c{i}, mapTransform(c{i}), e{i}, f{i});
%             fprintf(fid, '%s %s %s %s %s %s\n', a{i}, b{i}, c{i}, d{i}, e{i}, f{i});
        end

    end
    fclose(fid);


end