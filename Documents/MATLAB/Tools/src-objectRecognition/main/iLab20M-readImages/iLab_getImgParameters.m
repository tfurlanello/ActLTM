function imgPara = iLab_getImgParameters
% read camera, light, focus, rotation information of 20M images
    if nargin ~= 0
        imgPara = [];
        return;
    end    
    imgParaFile = '/lab/jiaping/svn-jiaping/projects/iLab-object-recognition/src/main/iLab20M-data-info/imagefiles-parameters.mat';
    imgPara = load(imgParaFile);

end