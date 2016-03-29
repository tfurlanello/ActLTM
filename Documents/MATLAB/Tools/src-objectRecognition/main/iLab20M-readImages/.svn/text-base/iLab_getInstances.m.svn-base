%% obsolete
function instances = iLab_getInstances(varargin)
    narginchk(1,1);
    dataroot = iLab_getRoot;
    classes = getSubfolders(dataroot);
    
    if isnumeric(varargin{1})
        if varargin{1} < 1 || varargin{1} > length(classes)
            fprintf(1, 'classIdx should be within (%d %d)\n', 1, length(classes));
            instances = [];
            return;
        end    
        instanceroot = fullfile(dataroot, classes{varargin{1}});
    elseif ischar(varargin{1})
        idx = strfind(classes, varargin{1});
        idx = ~cellfun('isempty', idx);
        if sum(idx) == 0
            fprintf(1, 'This class doesn''t exist\n');
            instances = [];
            return;
        else
            instanceroot = fullfile(dataroot, varargin{1});
        end
    end
    
    instances = getSubfolders(instanceroot);
    
end