function rootdir = rgbd_getDatasetRootDir(raw_crop)

    if ~exist('raw_crop', 'var') || isempty(raw_crop)
        raw_crop = 'raw';
    end
    
    switch raw_crop
        case 'raw'
            rootdir = '/lab/jiaping/svn-jiaping/projects/iLab-object-recognition/data/washington-RGBD-raw';
        case 'crop'            
            rootdir = '/lab/jiaping/svn-jiaping/projects/iLab-object-recognition/data/washington-RGBD-crop';
        otherwise
    end

end