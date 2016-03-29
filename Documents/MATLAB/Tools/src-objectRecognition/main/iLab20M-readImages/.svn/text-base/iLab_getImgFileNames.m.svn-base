function imageNames = iLab_getImgFileNames
   
    % return all 20M image names
    if nargin ~= 0
        imageNames = {};
        return;
    end
    
    imgNamesFile = '/lab/jiaping/svn-jiaping/projects/iLab-object-recognition/src/main/iLab20M-data-info/imagefiles-names.txt';
    fid         = fopen(imgNamesFile, 'r');
    imageNames  = textscan(fid, '%s\n');
    imageNames = imageNames{1};
    fclose(fid);

end