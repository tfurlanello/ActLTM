function b = iLab_imgExist(inputs)
    opts = iLab_validateImgFilePara;
    
    [opts, args] = vl_argparse(opts, inputs);
    
    imgRootDir              = iLab_getRoot;
    classFolderName         = opts.class;
    instanceFolderName      = iLab_genInstanceFolderName({'class', opts.class, 'instance', opts.instance});
    backgroundFolderName    = iLab_idx2nameBackgroundFolder(opts.background);
    
    imgFileName = iLab_genImgFileName(opts);
    imgFile = fullfile(imgRootDir, classFolderName, instanceFolderName, backgroundFolderName, imgFileName);

    if ~exist(imgFile, 'file')
        b = false;
    else
        b = true;
    end
    
end