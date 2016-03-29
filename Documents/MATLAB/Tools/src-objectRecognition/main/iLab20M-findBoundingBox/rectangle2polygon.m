function polygon = rectangle2polygon(rec, hole)
% 

    narginchk(1,2);
    if ~exist('hole', 'var') || isempty(hole)
        hole = false;
    end
    
    polygon = struct('x', [], ...
                     'y', [], ...
                     'hole', []);
                 
    
    xs = rec(1); 
    ys = rec(2);
    w = rec(3); h = rec(4);
    
    xe = xs + w - 1;
    ye = ys + h - 1;
    
    x = [xs xe xe xs];
    y = [ys ys ye ye];
    
    polygon.x = x;
    polygon.y = y;
    polygon.hole = hole;
end