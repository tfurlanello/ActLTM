function varargout = iLab_genLabelCR(args)
% generate labels from the given camera and rotation settings
    
    imgParams   = vl_argparse(iLab_validateImgFilePara, args);
    
    cameraRef   = imgParams.camera;
    rotationRef = imgParams.rotation;
    
    cameras = iLab_getCameras;
    rotations = iLab_getRotations;
    if ~ismember(cameraRef, cameras)
        error('camera labels: %s\n', mat2str(cameras));
        varargout = {'null'};
        return;
    end
    
    if ~ismember(rotationRef, rotations)
        error('rotation labels: %s\n', mat2str(rotations));
        varargout = {'null'};
        return;
    end
    
    labelCR = [iLab_idx2nameCamera(cameraRef) '-' iLab_idx2nameRotation(rotationRef)];
    
    if nargout == 0
        fprintf(1, 'label (camera: %d, rotation: %d): %s\n', ...
                        cameraRef, rotationRef, labelCR);
    else    
        varargout = {labelCR};
    end
end