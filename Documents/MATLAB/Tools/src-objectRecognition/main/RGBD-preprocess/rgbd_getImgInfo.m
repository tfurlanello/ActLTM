function imgInfo = rgbd_getImgInfo(raw_crop)

	if nargin >=2
        imgInfo = [];
        return;
    elseif nargin == 0
        raw_crop = 'raw';
    end    
    global workdir;
    
    switch raw_crop
        case 'crop'
            imgParaFile = fullfile(workdir, 'main', 'RGBD-data-info', 'imgInfo-crop.mat');
            load(imgParaFile, 'imgInfo');
        case 'raw'
            imgParaFile = fullfile(workdir, 'main', 'RGBD-data-info', 'imgInfo-raw.mat');
            load(imgParaFile, 'imgInfo');
        otherwise
    end
            

end