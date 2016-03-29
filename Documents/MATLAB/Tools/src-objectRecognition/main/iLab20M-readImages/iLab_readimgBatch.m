function varargout = iLab_readimgBatch(cameras, rotations, varargin)
    opts = iLab_validateImgFilePara;
    
    [opts, args] = vl_argparse(opts, varargin);
    
    imgRootDir              = iLab_getRoot;
    classFolderName         = opts.class;
    instanceFolderName      = iLab_genInstanceFolderName({'class', opts.class, 'instance', opts.instance});
    backgroundFolderName    = iLab_idx2nameBackgroundFolder(opts.background);
    
    imgFileName = iLab_genImgFileName(opts);
    imgFile = fullfile(imgRootDir, classFolderName, instanceFolderName, backgroundFolderName, imgFileName);
    if ~exist(imgFile, 'file')
        fprintf(1, 'file doesn''t exist\n');
        instances = iLab_retrieveInstances({'class', opts.class});
        fprintf(1, '%s has instances: %s\n', opts.class, ...
                        mat2str(instances));
 
        if find(instances == opts.instance)
            fprintf(1, '%s - %d is taken under backgrounds: %s\n', opts.class, opts.instance, ...
                 mat2str(iLab_retrieveBackgrounds({'class', opts.class, 'instance', opts.instance})));
        end
        
        varargout = {[], imgFileName, args};
        return;
    end
%     im = imread(imgFile);     
%     varargout = {im, imgFileName, args};
    
    [nNeighborCameras, nNeighborRotations]  = size(cameras);
    ims = [];
%     imgFolder = '/lab/jiaping/projects/iLab-object-recognition/src/main/iLab20M-splitImages/splitImgs';
    for c=1:nNeighborCameras
        for r=1:nNeighborRotations
            opts.camera = cameras(c,r);
            opts.rotation = rotations(c,r);
            imgFileName = iLab_genImgFileName(opts);
            imgFile = fullfile(imgRootDir, classFolderName, instanceFolderName, backgroundFolderName, imgFileName);
%             imgFile = fullfile(imgFolder, imgFileName);
            im = imread(imgFile);
            ims = cat(2, ims, im(:));
        end
    end
    
    im = imCollage(ims, [size(im,1) size(im,2)], [nNeighborRotations, nNeighborCameras]);
    if nargout == 0
        figure; imshow(im);
    else
        varargout = {im,ims};
    end
    
end