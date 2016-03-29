function neighbors = iLab_getNeighboringCR4(args)
% get 4-neighbors of the reference camera & rotation
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
  
    % (1) get up-down neighbors
    if cameraRef == 0
        neighbors_ud = 1;
    elseif cameraRef == 10
        neighbors_ud = 9;
    elseif cameraRef == 4
        neighbors_ud = 3;
    elseif cameraRef == 5
        neighbors_ud = 6;
    else
        neighbors_ud = [cameraRef-1 cameraRef + 1];
    end
    
    ud = struct('camera', {}, ...
                'rotation', {});
    for i=1:numel(neighbors_ud)
        ud(i).camera = neighbors_ud(i);
        ud(i).rotation = rotationRef;
    end

    % (2) get left-right neighbors
    if cameraRef ~= 5
        [nei_cameras, nei_rotations] =  iLab_getNeighboringRotations(1,...
                        {'camera', cameraRef, 'rotation', rotationRef});
        nei_cameras = nei_cameras([1 3]);
        nei_rotations = nei_rotations([1 3]);
    else
        if rotationRef == 7
            nei_cameras = 5;
            nei_rotations = 6;
        elseif rotationRef == 0
            nei_cameras = 5;
            nei_rotations = 1;
        else
            nei_cameras = [5 5];
            nei_rotations = [rotationRef-1 rotationRef+1];
        end
    end
    
    lr = struct('camera', {}, ...
                'rotation', {});
    for i=1:numel(nei_cameras)
        lr(i).camera = nei_cameras(i);
        lr(i).rotation = nei_rotations(i);
    end
    
    
    neighbors.ud = ud;
    neighbors.lr = lr;


end