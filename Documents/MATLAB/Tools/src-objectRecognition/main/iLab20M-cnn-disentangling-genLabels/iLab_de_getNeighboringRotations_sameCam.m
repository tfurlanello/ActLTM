function MapNeighboringRotations = iLab_de_getNeighboringRotations_sameCam(nNeighbors)
    
    if ~exist('nNeighbors', 'var') || isempty(nNeighbors)
        nNeighbors = 2;
    end
    
    switch nNeighbors
        case 2
            rotRef = [  0      1      2      3      4       5      6 ];
            rotNei = {[1 2], [2 3], [3 4], [4 5], [5 6],  [6 7],   7};
        case 1
            
            rotRef = [  0      1      2      3      4       5      6 ];
            rotNei = rotRef + 1;
            
        otherwise
            
    end    
    MapNeighboringRotations =  containers.Map(rotRef, rotNei);
end