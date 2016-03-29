function varargout =  iLab_getNeighboringRotations(ext, args)
% retrieve the neighboring rotations of the reference rotation
% inputs:
%          ext - neighborhood size
%          args - settings of image parameters: two parameters will be used
%                 reference camera, and the rotation of this camera
% outputs:
%          neighbors - neighboring rotations of the reference rotation

    if ~exist('ext', 'var') || isempty(ext)
        ext = 1;
    end
%     ext = min(7,ext);
    
    imgParams       =   vl_argparse(iLab_validateImgFilePara, args);
    cameraRef       =   imgParams.camera;    
    rotationRef     =   imgParams.rotation; 
    rotations       =   iLab_getRotations;
    nRotations      =   numel(rotations);
    cameraComp      =	iLab_getComplementCamera(cameraRef);

    rotationRefIdx = find(rotationRef == rotations,1);
    if isempty(rotationRefIdx)
        fprintf(1, 'camera has rotations: %s\n', mat2str(rotations));
        varargout = {'null'};
        return;
    end
    
    % left rotations
    LeftRotations = zeros(ext,1);
    LeftCameras = zeros(ext, 1);
    bTransition = false;
    ro = rotationRefIdx;
    ca = cameraRef;
    for r=1:ext
        ro = ro  - 1;
        if ro == 0
            ro = nRotations;
            bTransition = ~bTransition;
        end
        if bTransition == false
            ca = cameraRef;
        else
            ca = cameraComp;
        end
        LeftRotations(r) = rotations(ro);
        LeftCameras(r) = ca;
    end
    % right rotations
	RightRotations = zeros(ext,1);
    RightCameras = zeros(ext,1);
    bTransition = false;
    ro = rotationRefIdx;
    ca = cameraRef;
    for r=1:ext
        ro =  ro + 1;
        if ro == (nRotations+1)
            ro = 1;
            bTransition = ~bTransition;        
        end
        if bTransition == false
            ca = cameraRef;
        else
            ca = cameraComp;
        end
        RightRotations(r) = rotations(ro);
        RightCameras(r) = ca;        
    end
    
    neighboringRotations = [LeftRotations(ext:-1:1); rotationRef; RightRotations];
    neighboringCameras   = [LeftCameras(ext:-1:1); cameraRef; RightCameras];
    
    if nargout ~= 0
        varargout = {neighboringCameras, neighboringRotations};
    else
        outMsg = [];
        for i=1:numel(neighboringCameras)
            outMsg = strcat(outMsg, ',', iLab_genLabelCR({'camera', neighboringCameras(i), ...
                                                'rotation', neighboringRotations(i)}));
        end        
        fprintf(1, 'neighbors of (camera %d, rotation %d): %s\n', cameraRef, rotationRef, outMsg);

    end
    
 
end