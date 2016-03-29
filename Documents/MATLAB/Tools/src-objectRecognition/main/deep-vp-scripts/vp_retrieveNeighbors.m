function neighbors =  vp_retrieveNeighbors(discreteMapSize, index)
% inputs
%           discreteMapSize - map size of the discretization
%           index       - label index

% ouputs 
%           neighbors - neighbors of the target index
    
    
    h = discreteMapSize(1);
    w = discreteMapSize(2);
    [ridx, cidx] = ind2sub(discreteMapSize, index);
    
    neighbors = [];
    for dr=-1:1
        rr = ridx+dr;
        if rr <= 0 || rr >= h
            continue;
        end
        for dc=-1:1
           rc = cidx+dc;
           if rc <= 0 || rc >= w
               continue;
           end
            
           neighbors = cat(1, neighbors, sub2ind(discreteMapSize, rr, rc));
            
        end
    end
        
    
end