function backgrounds = iLab_retrieveBackgrounds(args)

    imgPara = iLab_validateImgFilePara;
    opts = vl_argparse(imgPara, args);
    
    className = opts.class;  
    idx = strfind(iLab_getClasses, className);
    b = ~cellfun('isempty', idx);
    if sum(b) == 0
        error('The input class doesn''t exist\n');
    end
	imgHierarchy = iLab_getImgHierarchy;    
    
    instances = imgHierarchy.(className).instances;
    b = instances == opts.instance;
    if sum(b) == 0
        fprintf(1, 'instances: %s\n', mat2str(instances));
        error('The input instance doesn''t exist\n');
    end
    
    instanceName = iLab_idx2nameInstance(opts.instance);    
    backgrounds = imgHierarchy.(className).(instanceName).backgrounds;    

end