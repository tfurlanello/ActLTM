function varargout = iLab_getNeighboringCR(ext_camera, ext_rotation, args)
% return the neighboring cameras and rotations
% inputs:
%         ext_camera    - neighborhood size: camera
%         ext_rotation  - neighborhood size: rotation
%         args          - image parameters, including camera and rotation
% outputs:
%         neighborhood matrix
%         
    imgParams   = vl_argparse(iLab_validateImgFilePara, args);
    cameraRef   = imgParams.camera;
    rotationRef = imgParams.rotation;
    
    neighborMatCameras   = [];
    neighborMatRotations = [];

    
    neighborCameras = iLab_getNeighboringCameras(cameraRef, ext_camera);
    nNeighborCamera = numel(neighborCameras);
    
    for c=1:nNeighborCamera        
             [neighborMatCamera, neighborMatRotation] = ...
                iLab_getNeighboringRotations(ext_rotation, {'camera', neighborCameras(c), ...
                                                             'rotation', rotationRef});
            neighborMatCameras = cat(1, neighborMatCameras, neighborMatCamera');
            neighborMatRotations = cat(1, neighborMatRotations, neighborMatRotation');
    end
    
    varargout = {neighborMatCameras, neighborMatRotations};
    
    
end