function ImageNet_cropTestImages(cropsize)
    

    if ~exist('cropsize', 'var') || isempty(cropsize)        
       cropsize = 256; 
    end
        
    testdir = ImageNet_getTestImagesDir;
    imExt = '.JPEG';
    
    idx = strfind(testdir, filesep);
    savedir = fullfile(testdir(1:idx(end)-1), ['test-crop-' num2str(cropsize)]);
    if ~exist(savedir, 'dir')
        mkdir(savedir);
    end
    
    imgs = dir(fullfile(testdir, ['*' imExt]));
    
    nImgs = numel(imgs);
    assert(nImgs == 150000);
    
    parfor i=1:nImgs
       if rem(i,1000) == 0
           i
       end
       i_img = imread(fullfile(testdir, imgs(i).name));       
       im = iLab_imrescale(i_img, [cropsize cropsize]);        
       imwrite(im, fullfile(savedir, imgs(i).name));        
    end
    
    imNames = {imgs.name};
    imNames = sort(imNames);
    labels = ImageNet_getTestGroundTruth;
    
    meta = ImageNet_getMetaData;
    
    fid = fopen(fullfile(savedir, 'test-check.txt'), 'w');
    for i=1:nImgs
        fprintf(fid, '%s %d %s\n', imNames{i}, labels(i), meta(labels(i)).words);
    end
    fclose(fid);
    
    
    fid = fopen(fullfile(savedir, 'test.txt'), 'w');
    for i=1:nImgs
        fprintf(fid, '%s %d\n', imNames{i}, labels(i));
    end
    fclose(fid);
    
    
end