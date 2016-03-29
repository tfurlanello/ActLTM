function imgNames = getImgFiles(parentDir, ext)
    
    imgFiles = dir(fullfile(parentDir, ['*', ext]));
	isub = [imgFiles(:).isdir]; %# returns logical vector
    imgNames = {imgFiles(~isub).name}';     
    
    % make sure the returned file names are the same under different runs
    imgNames = sort(imgNames);

end