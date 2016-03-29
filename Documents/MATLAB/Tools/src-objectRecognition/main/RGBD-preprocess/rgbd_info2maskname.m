function imName =  rgbd_info2maskname(imInfo)
    
    

    b = ismember({'class', 'instance', 'camera', 'frame'}, fieldnames(imInfo));
    if ~all(b)
        error('only support naming convention of rgbd dataset\n');
    end
    
    ext = 'maskcrop.png';
    us = '_';
    category = imInfo.class;
    instance = imInfo.instance;
    camera   = imInfo.camera;
    frame    = imInfo.frame;
    
    imName = [category us num2str(instance) us num2str(camera) ...
                us num2str(frame) us ext];

    
        
    
end