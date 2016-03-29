function neighboringRotation = iLab_de_getNeighboringRotation(r)
    
    neighboringRotation = r + 1 ;
    rotations = iLab_getRotations;
    
    r_min = rotations(1);
    r_max = rotations(end);
    
    if neighboringRotation > r_max || ...
            neighboringRotation < r_min
        neighboringRotation = [];
    end
    
end