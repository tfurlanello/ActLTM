function instances = iLab_retrieveInstances(args)
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
    
end