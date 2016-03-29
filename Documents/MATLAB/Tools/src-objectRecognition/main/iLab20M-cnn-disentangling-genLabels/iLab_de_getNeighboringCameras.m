function neighboringCameras = iLab_de_getNeighboringCameras(skip)
    
    narginchk(0,1);
    if ~exist('skip', 'var') || isempty(skip)
        skip = 2;
    end

    switch skip
        case 1
            neighboringCameras = [0 1;  
                                  1 2;  
                                  2 3;  
                                  3 4; 
                                  5 6;
                                  6 7;
                                  7 8;
                                  8 9;
                                  9 10];
        case 2
            
            neighboringCameras = [0 2;
                                  1 3;
                                  2 4;
                                  5 7;
                                  6 8;
                                  7 9;
                                  8 10];
            
        otherwise
            error('now only support skip 1 or 2\n');
    end
    
    
    


end