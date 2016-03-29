function varargout = rgbd_readmask(arg, raw_crop)
    
    if ~exist('raw_crop', 'var') || isempty(raw_crop)
        raw_crop = 'raw';
    end

    opts = rgbd_validateImgFilePara;    
    [opts, args] = vl_argparse(opts, arg);
    
    imgRootDir              = rgbd_getDatasetRootDir(raw_crop);
    classFolderName         = opts.class;
    instanceFolderName      = rgbd_genInstanceFolderName({'class', opts.class, 'instance', opts.instance});
    
    imgFileName = rgbd_genMaskFileName(opts, raw_crop);
            
    imgFile = fullfile(imgRootDir, classFolderName, instanceFolderName, imgFileName);
    if ~exist(imgFile, 'file')
        fprintf(1, 'file doesn''t exist\n');
        instancesID = rgbd_retrieveInstancesID({'class', opts.class});
        instances = rgbd_retrieveInstances({'class', opts.class});
        fprintf(1, '%s has instances: %s\n', opts.class, ...
                        mat2str(instancesID));
        
        if ismember(instanceFolderName, instances) && ~ismember(opts.camera, rgbd_getCameraIdx)
            fprintf(1, 'instance ''%s'' of %s is taken under cameras %s\n', ...
                            instanceFolderName, opts.class, mat2str(rgbd_getCameraIdx));
        end
        
        varargout = {[], imgFileName, args};
        return;
    end
    im = imread(imgFile);   
    
%     if nargout == 0
%         figure; imshow(im);
%         title(imgFileName);
%     else    
          varargout = {im, imgFileName, args};
%     end
end