function neighbors = iLab_getNeighboringCR5(args)
% not only get 4 neighbors, in terms of rotations and tilts
% but also set the opposite camera as its neighbors

    opts = struct('camera', 4, 'rotation', 3);
    imgParams       =   vl_argparse(opts, args);
    cameraRef       =   imgParams.camera;    
    rotationRef     =   imgParams.rotation;    

	if ~isscalar(cameraRef)
        error('only support scalar inputs\n');
    end
    cameras = iLab_getCameras;
    if ~ismember(cameraRef, cameras)
        error('cameras indices are: %s\n', mat2str(cameras));
    end
    
    rotations       =   iLab_getRotations;
    rotationRefIdx = find(rotationRef == rotations,1);
    if isempty(rotationRefIdx)
        error('camera has rotations: %s\n', mat2str(rotations));        
    end
    
    % 4 neighbors
	neighbors4 = iLab_getNeighboringCR4(imgParams);
    % set the opposite camera under the same rotation as its neighbors as
    % well
    opposite = struct('camera', {}, ...
                'rotation', {});
    if cameraRef ~= 5
        opposite(1).camera = 10-cameraRef;
        opposite(1).rotation = rotationRef;    
    end
    
    neighbors = neighbors4;
    neighbors.opposite = opposite;


end