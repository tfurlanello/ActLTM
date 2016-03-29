function [labelsCR, weights] = iLab_weightingNeighborsCR5(args)
    % 5 neighbors: 4 neighbors + 1 opposite 
	% input validity check
    
    % default weight
    ref         =   0.3;
    opposite    =   0.2;
    tot_ud      =   0.7;
    tot_lr      =   0.4;    
    
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
    neighbors = iLab_getNeighboringCR5(args);
    

    
    labelsCR = {};
    weights = [];
    labelsCR = cat(1, labelsCR, iLab_genLabelCR({'camera', cameraRef, 'rotation', rotationRef}));
     
    if numel(neighbors.opposite) == 0
        weights = cat(1, weights, ref+opposite);
    else
        labelsCR = cat(1, labelsCR, iLab_genLabelCR({'camera', neighbors.opposite(1).camera, ...
                                                    'rotation', neighbors.opposite(1).rotation}));
        
        weights = cat(1, weights, [ref; opposite]);
 
    end
    
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