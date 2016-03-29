function instances = rgbd_retrieveInstances(args, raw_crop)
    
    if ~exist('raw_crop', 'var') || isempty(raw_crop)
        raw_crop = 'raw';
    end

    imgPara = rgbd_validateImgFilePara;
    opts = vl_argparse(imgPara, args);
    
    className = opts.class;  
    b = ismember(className, rgbd_getClasses);    
    if ~b
        error('The input class doesn''t exist\n');
    end
    
    imgHierarchy = rgbd_getImgHierarchy(raw_crop);    
    instances = imgHierarchy.(className).instances;    
    
end