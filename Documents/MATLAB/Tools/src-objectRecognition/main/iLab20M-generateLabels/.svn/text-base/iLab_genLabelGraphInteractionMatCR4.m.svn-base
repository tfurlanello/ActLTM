function interactionMat =  iLab_genLabelGraphInteractionMatCR4(labelsCR)
    
    cameras = iLab_getCameras;
    rotations = iLab_getRotations;
    
    nCameras = numel(cameras);
    nRotations = numel(rotations);
    
    nLabels = numel(labelsCR);
    
    if ~isequal(nLabels, nCameras * nRotations)
        error('Input labels are not consistent with iLab20M dataset\n');
    end

    keySetCam      =  labelsCR;
    valueSetCam    =  1:numel(keySetCam);
    mapLabelsCam   =  containers.Map(keySetCam,valueSetCam); 
   
    interactionMat = zeros(nLabels, nLabels);
    
    for c=1:nCameras
        for r=1:nRotations
            ref_Label = iLab_genLabelCR({'camera', cameras(c), 'rotation', rotations(r)});
            [nei_labels, weights] = ...
                     iLab_weightingNeighborsCR4({'camera', cameras(c), 'rotation', rotations(r)});
            
            ref_idx = mapLabelsCam(ref_Label);
            for n=1:numel(nei_labels)
                tar_idx = mapLabelsCam(nei_labels{n});
                interactionMat(ref_idx, tar_idx) = weights(n);
            end
            
        end
    end
        
    
end