function imgInformation = rgbd_buildImgInfo(raw_crop)
        
    if ~exist('raw_crop', 'var') || isempty(raw_crop)
        raw_crop = 'raw';
    end

    rootdir = rgbd_getDatasetRootDir(raw_crop);
    categories = rgbd_getClasses;
    
    nCategories = numel(categories);    
    cnt = 0;
    % img: _crop.png
    for c=1:nCategories
       c_class = categories{c};
       cand_instances = dir(fullfile(rootdir, c_class));
       instances = {};
       for i=1:numel(cand_instances)
           if cand_instances(i).isdir && ~ismember(cand_instances(i).name, {'.', '..'})
               instances = cat(2, instances, cand_instances(i).name);
           end
       end
       
       nInstances = numel(instances);
       
       for i=1:nInstances
           fprintf(1, 'processing: category %s (%d/%d), instance %s (%d/%d)\n', ...
                        c_class, c, nCategories, instances{i}, i, nInstances);
           i_instance = instances{i};
           i_dir = fullfile(rootdir, c_class, i_instance);
           
           % parse file name
           switch raw_crop
               case 'crop'
                   imPattern = '_crop.png';
                   imgs = dir(fullfile(i_dir, ['*' imPattern]));           
               case 'raw'
                   imPattern = '.png';
                   imgs = dir(fullfile(i_dir, ['*' imPattern]));           
                   imgsNames = {imgs.name};
                   raw_idx = regexp(imgsNames, ['\d' imPattern]);
                   nraw_idx = cellfun('isempty', raw_idx);
                   imgs = imgs(~nraw_idx);                   
               otherwise
           end
           nimgs = numel(imgs);           
           imInfo = struct('class', {}, 'instance', 1, 'camera', 1, 'frame', 1);           
           for n=1:nimgs
               
               switch raw_crop
                   case 'crop'
                        n_imInfo = rgbd_parsefilename(imgs(n).name);
                   case 'raw'
                        n_imInfo = rgbd_parsefilenameRawImg(imgs(n).name);
               end
               fnames = fieldnames(n_imInfo);
               cnt = cnt + 1;
               for f=1:numel(fnames)
                  imInfo(n).(fnames{f}) = n_imInfo.(fnames{f});
                  imgInformation(cnt).(fnames{f}) = n_imInfo.(fnames{f});
               end               
           end           

       end
       
    end
    imgInfo = imgInformation; clear imgInformation;
    global workdir;
    
    switch raw_crop
        case 'raw'
            save(fullfile(workdir, 'main', 'RGBD-data-info', 'imgInfo-raw.mat'), 'imgInfo');
        case  'crop'
            save(fullfile(workdir, 'main', 'RGBD-data-info', 'imgInfo-crop.mat'), 'imgInfo');
        otherwise
            
    end
            
  
end