function options =  iLab_parseFileName(fileName)
% this function is designed specifically for ilab-20M datasets
% fileName. It doesn't generalize to more general cases

    symbols = {'i', 'b', 'c', 'r', 'l', 'f'};
    lengths = [4, 4, 2, 2, 1, 1];
    names   = {'instance', 'background', 'camera', 'rotation', 'light', 'focus'};
    
    % first, get the class name (it should be appear at the front)
    classes = iLab_getClasses;
    idx = strfind(fileName, '-');
    if isempty(idx)
        error('make sure the file is named to the convention of iLab20M datast\n');
    end
    class = fileName(1:idx(1)-1);
    if ~ismember(class, classes)
        error('make sure the class name is in the front\n');
    end
    options.class = class;
    
    substring = fileName(idx(1)+1:end);
    
    nSymbols = numel(symbols);
    
    for s=1:nSymbols
        idx = strfind(substring, symbols{s});
        if isempty(idx)
            continue;
        elseif numel(idx) >= 2
            error('only support iLab-20M style file name\n');
        end        
        options.(names{s}) = str2double(substring(idx+1:idx+lengths(s)));        
    end
    
     
end