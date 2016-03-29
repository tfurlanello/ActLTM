function MapNeighboringCameras = iLab_de_getNeighboringCameras_sameRot
    
    assert(nargin == 0);
    
    camRef = [  0      1      2    3    5      6      7      8     9];
    camNei = {[1 2], [2 3], [3 4], 4, [6 7], [7 8], [8 9], [9 10], 10};
    
    
    MapNeighboringCameras =  containers.Map(camRef, camNei);
end