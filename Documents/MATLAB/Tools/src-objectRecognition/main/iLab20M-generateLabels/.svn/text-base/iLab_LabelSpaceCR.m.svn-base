function mapLabels =  iLab_LabelSpaceCR()
    % creat a map object, which maps keys to values
    % i.e., keys: labels of the images
    %       values: index used in the model
    cameras     =   iLab_getCameras;
    rotations   =   iLab_getRotations;
    
    nCameras    =   numel(cameras);
    nRotations  =   numel(rotations);
    
    nLabels     =   nCameras * nRotations;
    
    labels_CR   =   cell(nLabels,1);
    
    cnt = 0;
    for c=1:nCameras
         for r=1:nRotations
             cnt = cnt + 1;
            labels_CR{cnt} = iLab_genLabelCR({'camera', cameras(c), 'rotation', rotations(r)});
        end
    end
    
    keySet      = labels_CR;
    valueSet    = 1:cnt;
    mapLabels   = containers.Map(keySet,valueSet); 
    
end