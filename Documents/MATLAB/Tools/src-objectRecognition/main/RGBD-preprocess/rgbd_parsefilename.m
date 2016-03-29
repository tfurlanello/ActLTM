function imInfo =  rgbd_parsefilename(filename)
    
    imInfo = struct('class', {}, ...
                    'instance', [], ...
                    'camera', [], ...
                    'frame', []);
                
    idx = strfind(filename, '_');
    try
        assert(numel(idx) >= 4);
    catch
        error('only support the naming convention in RGB-D dataset\n');
    
    end
    
    imInfo(1).class     = filename(1:idx(end-3)-1);
    imInfo(1).instance  = str2double(filename(idx(end-3)+1:idx(end-2)-1));
    imInfo(1).camera    = str2double(filename(idx(end-2)+1:idx(end-1)-1));
    imInfo(1).frame     = str2double(filename(idx(end-1)+1:idx(end)-1));
    
    
    
end