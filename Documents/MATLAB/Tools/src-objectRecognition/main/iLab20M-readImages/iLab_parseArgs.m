function options = iLab_parseArgs(varargin)

    % Parse the input arguments
    % Set the defaults

    options = struct('class',       'van', ...
                     'bclass',      false, ... 
                     'instance',    1, ...
                     'binstance',   false, ...
                     'background',  1, ...
                     'bbackground', false, ...
                     'camera',      1, ... 
                     'bcamera',     false, ...
                     'rotation',    1, ...
                     'brotation',   false, ...
                     'lighting',     0, ...
                     'blighting',   false, ...
                     'focus',       1, ...
                     'bfocus',      false);

    n = length(varargin(:));
    if rem(n,2) ~= 0
        error('Even number of inputs\n');
        return;
    end

    for i=1:2:(n-1)
        if ischar(varargin{i})
            switch lower(varargin{i})
                case 'class'
                    options.class       = varargin{i+1};
                    options.bclass      = true;
                case 'instance'
                    options.instance    = varargin{i+1};
                    options.binstance   = true;
                case 'background'
                    options.background  = varargin{i+1};
                    options.bbackground = true;
                case 'camera'
                    options.camera      = varargin{i+1};
                    options.bcamera     = true;
                case 'rotation'
                    options.rotation    = varargin{i+1};
                    options.brotation   = true;
                case 'lighting'
                    options.lighting    = varargin{i+1};
                    options.blighting   = true;
                case 'focus'
                    options.focus       = varargin{i+1};
                    options.bfocus      = true;
            end
        end
    end

 
end