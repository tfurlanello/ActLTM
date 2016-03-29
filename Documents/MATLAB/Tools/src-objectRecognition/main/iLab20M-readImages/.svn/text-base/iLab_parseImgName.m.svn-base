function options =  iLab_parseImgName(fileName)
% this function is designed specifically for ilab-20M datasets
% fileName: doesn't have extension

    idx = regexp(fileName, '-');
    if length(idx) == 7
        idx = [0 idx(1:7)];
    elseif length(idx) == 6
        idx = [0 idx(1:6) length(fileName)+1];
    else
        error('Only support image file names defined by ilab-20M dataset\n');
    end    
    
    for i=1:7        
        str = fileName((idx(i)+1):(idx(i+1)-1));

        switch i
            case 1
                options.class = str;
            otherwise
                switch str(1)
                    case 'i'
                        options.instance    = str2double(str(2:end));
                    case 'b'
                        options.background  = str2double(str(2:end));
                    case 'c'
                        options.camera      = str2double(str(2:end));
                    case 'r'
                        options.rotation    = str2double(str(2:end));
                    case 'l'
                        options.lighting    = str2double(str(2:end));
                    case 'f'
                        options.focus       = str2double(str(2:end));
                end
        end
        
    end
    
end