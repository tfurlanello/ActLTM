function varargout = iLab_getNeighboringCameras(cameraRef, ext)
% retrieve the neighboring cameras 
% inputs:
%           idx - index of cameras
%           ext - neighborhood size (default 1)
% outputs:
%           neighbors of reference cameras

    narginchk(1,2);
    if ~exist('ext', 'var') || isempty(ext)
        ext = 1;
    end
    if ~isscalar(cameraRef)
        error('only support scalar inputs\n');
    end
    cameras = iLab_getCameras;
    nCameras = numel(cameras);
    if ~ismember(cameraRef, cameras)
        fprintf(1, 'cameras indices are: %s\n', mat2str(cameras));
        varargout = {'null'};
        return;
    end

    cameraRefIdx = find(cameraRef == cameras,1);    
    ranges      = (cameraRefIdx-ext):1:(cameraRefIdx+ext);
    neighbors   = intersect(ranges,1:nCameras);
    neighbors   = cameras(neighbors);
%     neighbors   = setdiff(cameras(neighbors), cameraRef);
 
    if nargout ~= 0
        varargout = {neighbors};
    else
        camera_reference = iLab_idx2nameCamera(cameraRef);
        cameras_neighbors = '';
        for c=1:numel(neighbors)
            cameras_neighbors = cat(2,...
                 cameras_neighbors, ' ', iLab_idx2nameCamera(neighbors(c)));
        end
        fprintf(1, 'Camera ''%s'' has neighbors: %s\n', ...
            camera_reference,  cameras_neighbors );
    end
    
    
end