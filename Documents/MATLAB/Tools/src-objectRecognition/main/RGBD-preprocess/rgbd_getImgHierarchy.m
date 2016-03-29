function imgHierarchy = rgbd_getImgHierarchy(raw_crop)

	if nargin >=2
        imgHierarchy = [];
        return;
    elseif nargin == 0
        raw_crop = 'raw';
    end
    
    global workdir;
    switch raw_crop    
        case 'crop'
            imgParaFile = fullfile(workdir, 'main', 'RGBD-data-info', 'imgHierarchy-crop.mat');
            load(imgParaFile, 'imgHierarchy');
        case 'raw'
            imgParaFile = fullfile(workdir, 'main', 'RGBD-data-info', 'imgHierarchy-raw.mat');
            load(imgParaFile, 'imgHierarchy');
        otherwise
    end
    
end