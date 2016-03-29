function MapNeighboringCameras = iLab_de_getNeighboringCameras_diffRot
    
    assert(nargin == 0);
    
    camRef = [  0      1      2      3    4  5      6      7      8      9     10];
    camNei = {  1,   [0 2], [1 3], [2 4], 3, 6,   [5 7], [6 8], [7 9], [8 10]  9};
    
    
    MapNeighboringCameras =  containers.Map(camRef, camNei);
end