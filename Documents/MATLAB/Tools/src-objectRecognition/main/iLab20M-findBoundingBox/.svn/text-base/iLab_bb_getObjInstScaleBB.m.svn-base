% 
function [objInstNames, scales, objInstCRnames, BBs] =  iLab_bb_getObjInstScaleBB

    categories = iLab_getClasses;
    nCategories = numel(categories);
    
    global   pathBB;
    resDir = pathBB.round1.res;
% 	resDir2 = pathBB.round2.res;
	resDir2 = pathBB.round3.res;
    error_lists = textread(fullfile(pathBB.round1.outlies, 'lists.collage'), '%s\n');
    
    %% case 1: each instance has only one scale
    scales       = [];
    objInstNames = {};
    fsep = '-';
    for c=1:nCategories
        fprintf(1, '1 processing: class - %s\n',categories{c});
        cClass = categories{c};
        instances = iLab_retrieveInstances({'class', cClass});
        nInstances = numel(instances);

        current_class = cClass;
        for i=1:nInstances
            current_instance = instances(i);

            [objw_med,objh_med, objw_max, objh_max] =  ...
                       iLab_bb_getObjInstScale({'class', current_class, 'instance', current_instance});
            [objw_relax,objh_relax] = iLab_bb_getObjInstScaleRelax({'class', current_class, 'instance', current_instance});

            
            name  = [current_class, fsep, iLab_idx2nameInstance(current_instance)];
            if isempty(objw_max) || isempty(objh_max) || ...
                    isempty(objw_relax) || isempty(objh_relax)
                scales = cat(1, scales, [256 256 256]);
            else
                tmpScale1 = max([objw_max objh_max]);
                tmpScale2 = max([objw_relax objh_relax]);
                objScale = tmpScale1 + round(4.0*(tmpScale2-tmpScale1)/5.0);
                
                objScale =  min([objScale 720]); % the maximal size is 720
                scales   =  cat(1, scales, [objScale tmpScale1 tmpScale2]);
            end
            objInstNames = cat(1, objInstNames, name);
            
        end
    end
    
    
    %% case 2: each instance has different bounding boxes under 
    %% different camera and rotations
    BBs = [];
    objInstCRnames = {};
    cameras = iLab_getCameras;
    rotations = iLab_getRotations;
    for cate=1:nCategories
        fprintf(1, '2 processing: class - %s\n',categories{cate});

        cClass = categories{cate};
        instances = iLab_retrieveInstances({'class', cClass});
        nInstances = numel(instances);

        current_class = cClass;
        opts.class = current_class;
        for i=1:nInstances
            current_instance = instances(i);
            opts.instance = current_instance;

            %% 1st round to detect outlies
            bbs = [];
            tmpNames = {};
            for c=1:numel(cameras)
                opts.camera = cameras(c);
                for r=1:numel(rotations)
                    opts.rotation = rotations(r);
                    BBfile  = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
                        fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation)];

                    if all(cellfun('isempty', strfind(error_lists, [BBfile '.jpg'])) )
                        if exist(fullfile(resDir, [BBfile '.bb']), 'file')
                            bb = dlmread(fullfile(resDir, [BBfile '.bb']));  
                        else
                            bb = [];
                            BBfile = {};
                        end
                    else
                        if exist(fullfile(resDir2, [BBfile '.bb']), 'file')
                            bb = dlmread(fullfile(resDir2, [BBfile '.bb']));  
                        else
                            bb = [];
                            BBfile = {};
                        end
                    end
                    
                    bbs = cat(1, bbs, bb);
                    tmpNames = cat(1, tmpNames, BBfile);
%                     BBs = cat(1, BBs, bb);
%                     objInstCRnames = cat(1, objInstCRnames, BBfile);
                end        
            end               
            %% manually define the bounding box of outlies
            if ~isempty(bbs) && ~isempty(tmpNames)
                BBs = cat(1, BBs, [bbs(:,1) bbs(:,2) bbs(:,3)+bbs(:,1)-1 bbs(:,4)+bbs(:,2)-1]);
                objInstCRnames = cat(1, objInstCRnames, tmpNames);
            end
            
        end
    end    
     

end