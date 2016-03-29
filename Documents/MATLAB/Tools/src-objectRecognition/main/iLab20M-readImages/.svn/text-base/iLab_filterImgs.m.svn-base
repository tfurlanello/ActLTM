function filteredIdx =  iLab_filterImgs(imgfilelist, varargin)
    
    % default option settings    
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
                 
     options = iLab_parseArgs(varargin);
     
     nimgFiles   = length(imgfilelist);
     fclass      = zeros(nimgFiles,1) < 1.0;
     finstance   = zeros(nimgFiles,1) < 1.0;
     fbackground = zeros(nimgFiles,1) < 1.0;
     fcamera     = zeros(nimgFiles,1) < 1.0;
     frotation   = zeros(nimgFiles,1) < 1.0;
     flighting   = zeros(nimgFiles,1) < 1.0;
     ffocus      = zeros(nimgFiles,1) < 1.0;
    
     % filter by class
     if options.bclass
         pattern = options.class;
         idx = strfind(imgfilelist, pattern);
         fclass = ~cellfun('isempty', idx);
%          for i=1:nimgFiles
%              fclass(i) = ~isempty(idx{i});
%          end
     end        
     
     % filter by instance
     if options.binstance
         pattern = iLab_idx2nameInstance(options.instance);
         idx = strfind(imgfilelist, pattern);
         finstance = ~cellfun('isempty', idx);

     end        
     
     % filter by background
     if options.bbackground
         pattern = iLab_idx2nameBackground(options.background);
         idx = strfind(imgfilelist, pattern);
         fbackground = ~cellfun('isempty', idx);

     end        
     
     % filter by camera
     if options.bcamera
         pattern = iLab_idx2nameCamera(options.camera);
         idx = strfind(imgfilelist, pattern);
         fcamera = ~cellfun('isempty', idx);

     end             
     
     % filter by rotation
     if options.brotation
         pattern = iLab_idx2nameRotation(options.rotation);
         idx = strfind(imgfilelist, pattern);
         frotation = ~cellfun('isempty', idx);

     end    
     
     % filter by lighting
     if options.blighting
         pattern = iLab_idx2nameLight(options.lighting);
         idx = strfind(imgfilelist, pattern);
         flighting = ~cellfun('isempty', idx);

     end        
     
     % filter by focus
     if options.focus
         pattern = iLab_idx2nameFocus(options.focus);
         idx = strfind(imgfilelist, pattern);
         ffocus = ~cellfun('isempty', idx);

     end     
     
     filteredIdx = fclass & finstance & fbackground & ...
                        fcamera & frotation & flighting & ffocus;        
    
end