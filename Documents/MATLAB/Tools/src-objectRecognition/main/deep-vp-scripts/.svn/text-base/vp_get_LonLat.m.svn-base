
function [lon, lat] = vp_get_LonLat(filename)
    p1 = '_lon_'; 
    p2 = '_lat_';
    p3 = '_pitch_';
    
    i1 = strfind(filename, p1);
    i2 = strfind(filename, p2);
    i3 = strfind(filename, p3);
    
    lon =  filename(i1+numel(p1):i2-1);
    lat =  filename(i2+numel(p2):i3-1);

end

