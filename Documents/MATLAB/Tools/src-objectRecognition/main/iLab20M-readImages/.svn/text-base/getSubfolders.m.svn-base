
function subFolders = getSubfolders(parentDir)

    d = dir(parentDir);
    isub = [d(:).isdir]; %# returns logical vector
    subFolders = {d(isub).name}'; 
    subFolders(ismember(subFolders,{'.','..'})) = [];
    
    % make sure the returned results are the same under different runs
    subFolders = sort(subFolders);

end