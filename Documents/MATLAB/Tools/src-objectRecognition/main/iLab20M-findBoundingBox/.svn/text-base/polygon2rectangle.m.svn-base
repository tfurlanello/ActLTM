function bb =  polygon2rectangle(poly)
    
    x = poly.x;
    y = poly.y;
    
    xs = min(x);
    xe = max(x);
    
    ys = min(y);
    ye = max(y);
    
    w = xe - xs + 1;
    h = ye - ys + 1;
    
%     assert( sum(x == xs) == 2);
%     assert( sum(x == xe) == 2);
%     
%     assert( sum(y == ys) == 2);
%     assert( sum(y == ye) == 2);
    
    bb = [xs ys w h];
    
    
end