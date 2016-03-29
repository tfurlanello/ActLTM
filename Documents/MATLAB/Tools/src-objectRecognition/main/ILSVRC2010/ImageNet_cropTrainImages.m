function [imgNames, labels] = ImageNet_cropTrainImages(cropsize, trainsize)

    if ~exist('cropsize', 'var') || isempty(cropsize)        
       cropsize = 256; 
    end
    
    if ~exist('trainsize', 'var') || isempty(trainsize)
       trainsize = 40; 
    end
    
    ncategories = 1000;
    
    traindir = ImageNet_getTrainImagesDir;
    imExt = '.JPEG';
    
    idx = strfind(traindir, filesep);
    savedir = fullfile(traindir(1:idx(end)-1), ['train-crop-' num2str(cropsize)]);
    if ~exist(savedir, 'dir')
        mkdir(savedir);
    end
    
    metadata = ImageNet_getMetaData;
    
    imgNames = {};
    labels = [];
    
    for c=1:ncategories
       c
       subTraindir = fullfile(traindir, metadata(c).WNID);
        
       imgs = dir(fullfile(subTraindir, ['*' imExt]));
       nimgs = numel(imgs);
       assert(nimgs > trainsize);
       
       rorder = randperm(nimgs);       
       for i=1:trainsize   
           try
               i_img = imread(fullfile(subTraindir, imgs(rorder(i)).name));
           catch
               continue;
           end
           im        =  iLab_imrescale(i_img, [cropsize cropsize]);           
           imwrite(im,  fullfile(savedir, imgs(rorder(i)).name));
           imgNames  =  cat(1, imgNames, imgs(rorder(i)).name);
           labels    =  cat(1, labels, c);
       end
    end
    
    fid = fopen(fullfile(savedir, 'train.txt'), 'w');
   
    for i=1:numel(imgNames)
       fprintf(fid, '%s %d\n', imgNames{i}, labels(i)); 
    end
    fclose(fid);
    
    
    meta = ImageNet_getMetaData;
	fid = fopen(fullfile(savedir, 'train-check.txt'), 'w');   
    for i=1:numel(imgNames)
       fprintf(fid, '%s %d %s\n', imgNames{i}, labels(i), meta(labels(i)).words); 
    end
    fclose(fid);
    
    
end