function imName =  rgbd_genImgFileName(args, raw_crop)
    
    if ~exist('raw_crop', 'var') || isempty(raw_crop)
        raw_crop = 'raw';
    end
    

    opts = rgbd_validateImgFilePara;
    imInfo = vl_argparse(opts, args);
    
    b = ismember({'class', 'instance', 'camera', 'frame'}, fieldnames(imInfo));
    if ~all(b)
        error('only support naming convention of rgbd dataset\n');
    end
    
    us = '_';
    category = imInfo.class;
    instance = imInfo.instance;
    camera   = imInfo.camera;
    frame    = imInfo.frame;
    
    
    switch raw_crop
        case 'raw'
            ext = '.png';
            imName = [category us num2str(instance) us num2str(camera) ...
                    us num2str(frame)  ext];
        case 'crop'
            ext = 'crop.png';
            imName = [category us num2str(instance) us num2str(camera) ...
                us num2str(frame) us ext];
        otherwise
    end
    
end