function  varargout = iLab_getComplementCamera(idx)
% return the camera which lies on on the same latitude with the reference
% camera
% inputs:
%           idx - index of a reference camera
% outputs:
%           cCamera - the complement of the reference camera
    
    if ~isscalar(idx)
        error('only support scalar inputs\n');        
    end
    
    cameras = iLab_getCameras;
    if ~ismember(idx, cameras)
        fprintf(1, 'cameras have indices: %s\n', mat2str(cameras));
        varargout = {'null'};
        return;
    end

    cCamera = max(cameras) - idx;
    
    if nargout == 0
        fprintf(1, 'reference camera: %s; its complement camera: %s\n', ...
                iLab_idx2nameCamera(idx), iLab_idx2nameCamera(cCamera));
    else
        varargout = {cCamera};
    end
    
end