
function classDir =  iLab_getClassRoot(varargin)  
    
    narginchk(1,1);
	rootdir     =   iLab_getRoot;
	classNames  =   iLab_getClasses;

    if isnumeric(varargin{1})    
        classDir    =   fullfile(rootdir, classNames{varargin{1}});
    elseif ischar(varargin{1})           
        idx = strfind(classNames, varargin{1});
        idx = ~cellfun('isempty', idx);
        if sum(idx) == 0
            fprintf(1, 'This class doesn''t exist\n');
            classDir = [];
            return;
        else
            classDir = fullfile(rootdir, varargin{1});
        end  
    end 
    
end