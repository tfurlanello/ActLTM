function center =  iLab_getCameraCenter(cameraIdx)

  X = [ 520, 520, 530, 512, 491, 466, 500, 490, 498, 503, 498 ];
  Y = [190, 151, 253, 330, 330, 356, 322, 299, 290, 207, 203];
  
  center = [X(cameraIdx) Y(cameraIdx)];
end