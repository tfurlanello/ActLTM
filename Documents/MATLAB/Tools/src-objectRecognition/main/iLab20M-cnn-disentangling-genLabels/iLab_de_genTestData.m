%% this script is used to imdb file for testing



dataDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/metadata';
trainImgsInfoFile  =  'trainImagesInfo.mat';
testImgsInfoFile   =  'testImagesInfo.mat';
classes = iLab_getClasses;

saveDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb';


mapDirs = {...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb/mappings-f7.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb/mappings-f11.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb/mappings-f18.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb/mappings-f56.mat'};

saveFilestxt = {'test-f7.txt', ...
            'test-f11.txt', ...
            'test-f18.txt', ...
            'test-f56.txt'};

saveFilesimdb = {'imdb-f7.mat', ...
    'imdb-f11.mat', ...
    'imdb-f18.mat', ...
    'imdb-f56.mat'};
        
nf = [7 11 18 56];        
        
        
load(fullfile(dataDir, testImgsInfoFile));
nTest = numel(testImagesInfo);
names = fieldnames(testImagesInfo);
values = [];
for f=1:numel(names)
    values = cat(1, values, cell2mat({testImagesInfo.(names{f})}));
end
    
 imgdataDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera'; 
   
    
for m=1:4

    load(mapDirs{m});

    %========================================================================== 
    %                                                           test images
    %==========================================================================
    ftest = fopen(fullfile(saveDir, saveFilestxt{m}), 'w');

    tic
    for i=1:nTest

        if rem(i, 1000) == 0
            i
            toc
        end

        ref_para     =  values(:,i);
        ref_camera   =  ref_para(4);
        ref_rotation =  ref_para(5);

        class = classes{ref_para(2)};
        ref_img_file = iLab_genImgFileName({'class', class, 'instance', ref_para(3), ...
            'background', ref_para(1), 'camera', ref_para(4), 'rotation', ref_para(5), ...
            'focus', ref_para(6), 'light', ref_para(7)});  

        transform = 'cr';
        transformIdx = 1;
        classIdx = mapObject(class);


        fprintf(ftest, '%s %d %s %d %s %s\n', class, classIdx, transform, ...
                transformIdx, ref_img_file, ref_img_file);        


    end   

    fclose(ftest);
    
    
    

      imdb = iLab_de_cnn_setupdata_tmp('dataDir', imgdataDir, 'lite', false, 'nf', nf(m)) ;
      save(fullfile(saveDir, saveFilesimdb{m}), '-struct', 'imdb') ;


end







