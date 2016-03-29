function [labelsCR, weights] = iLab_weightingNeighborsCR4(args)
	% input validity check
    
	ref = 0.6;
    tot_ud = 0.9;
    tot_lr = 0.7;
    
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

    % get neighbors
    neighbors = iLab_getNeighboringCR4(args);
    

    
    labelsCR = {};
    weights = [];
    labelsCR = cat(1, labelsCR, iLab_genLabelCR({'camera', cameraRef, 'rotation', rotationRef}));
    weights = cat(1, weights, ref);
    
    lr_weight = (tot_lr - ref)/numel(neighbors.lr);
    for i=1:numel(neighbors.lr)
        labelsCR = cat(1, labelsCR, iLab_genLabelCR({'camera', neighbors.lr(i).camera, ...
                                                'rotation', neighbors.lr(i).rotation}));
        weights = cat(1, weights, lr_weight);                                               
    end
    
    
    ud_weight = (tot_ud - ref)/numel(neighbors.ud);
    for i=1:numel(neighbors.ud)
        labelsCR = cat(1, labelsCR, iLab_genLabelCR({'camera', neighbors.ud(i).camera, ...
                                                'rotation', neighbors.ud(i).rotation}));
        weights = cat(1, weights, ud_weight);                                               
    end
    


end