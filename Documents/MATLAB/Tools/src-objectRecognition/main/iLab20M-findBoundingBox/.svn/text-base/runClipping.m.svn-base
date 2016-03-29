global pathBB;

[objInstNames, objInstScales, objInstSizes] = ...
    textread(fullfile(pathBB.round3.bb, 'iLab20M-obj-scales.txt'), '%s %d %d\n');
[objInstCRnames, x1,y1,x2,y2] = ...
    textread(fullfile(pathBB.round3.bb, 'iLab20M-obj-bb.txt'), '%s %d %d %d %d\n');

saveDir1 = fullfile(pathBB.round3.clipping, 'object-instance');
saveDir2 = fullfile(pathBB.round3.clipping, 'object-instance-camera-rotation');

if ~exist(saveDir1, 'dir')
    mkdir(saveDir1);
end

if ~exist(saveDir2, 'dir')
    mkdir(saveDir2);
end

cameras   = iLab_getCameras;
rotations = iLab_getRotations;
lights    = iLab_getLights;
nCameras     =  numel(cameras);
nRotations   =  numel(rotations);
nLights      =  numel(lights);


categories = iLab_getClasses;
nCategories = numel(categories);

fsep = '-';
rng('shuffle');
corder = randperm(nCategories);


%% case one
for cate=1:nCategories
    current_class = categories{corder(cate)};
    instances = iLab_retrieveInstances({'class', current_class});
    
    nInstances = numel(instances);
    iorder = randperm(nInstances);
    
    for ins=1:nInstances
        
        current_instance = instances(iorder(ins));

        instTag = [current_class, fsep, iLab_idx2nameInstance(current_instance)];
        
        if exist(fullfile(saveDir1, [instTag '.flag']), 'file')
            continue;
        end
        
        fid = fopen(fullfile(saveDir1, [instTag '.flag']), 'w');
        fprintf(fid,'1');
        fclose(fid);

%         tag = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
%                     fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation)];
%                
%                         
                
        backgrounds     = iLab_retrieveBackgrounds({'class', current_class, 'instance', current_instance});
        pureBackgrounds = [0 1 2 3 4 5 6];
        realbackgrounds = setdiff(backgrounds, pureBackgrounds);

        testBkg = realbackgrounds(randi(numel(realbackgrounds)));
        
        idx = find(strcmp(objInstNames, instTag));
        objScale  = objInstScales(idx);
        


        opts.class      = current_class;
        opts.instance   = current_instance;
        opts.focus      = 1;

        crops_square = [];
        opts.background = testBkg;
        for c=1:nCameras
            for l=1:nLights
                for r=1:nRotations
             
                    opts.light = lights(l);
                    opts.rotation = rotations(r);
                    opts.camera = cameras(c);
                    bImg = iLab_imgExist(opts);
                    if bImg == false
                        continue;
                    else
                        objCenter =  iLab_getCameraCenter(opts.camera+1);
                        im = iLab_readimg(opts);
                        [crop, objScale] = iLab_bb_intelligentClipper(im, objScale, objCenter);
                        crops_square = [crops_square crop(:)];
                    end          
                end
            end
        end

        if ~isempty(crops_square)
            tmp = imCollage(crops_square, [objScale objScale]);
            imwrite(tmp, fullfile(saveDir1, [instTag '.jpg']));
    	end 

    end
end


%% case two
for cate=1:0%nCategories
    current_class = categories{corder(cate)};
    instances = iLab_retrieveInstances({'class', current_class});
    
    nInstances = numel(instances);
    iorder = randperm(nInstances);
    
    for ins=1:nInstances
        
        current_instance = instances(iorder(ins));

        folderName = [current_class, fsep, iLab_idx2nameInstance(current_instance)];
        
        if exist(fullfile(saveDir2, [folderName '.flag']), 'file')
            continue;
        end
        
        fid = fopen(fullfile(saveDir2, [folderName '.flag']), 'w');
        fprintf(fid,'1');
        fclose(fid);

        if ~exist(fullfile(saveDir2, folderName), 'dir')
            mkdir(fullfile(saveDir2, folderName));
        end
                
        backgrounds     = iLab_retrieveBackgrounds({'class', current_class, 'instance', current_instance});
        pureBackgrounds = [0 1 2 3 4 5 6];
        realbackgrounds = setdiff(backgrounds, pureBackgrounds);

        testBkg = realbackgrounds(randi(numel(realbackgrounds)));
       


        opts.class      = current_class;
        opts.instance   = current_instance;
        opts.focus      = 1;

        opts.background = testBkg;
        for c=1:nCameras
            for l=1:nLights
                for r=1:nRotations
                    opts.light = lights(l);
                    opts.rotation = rotations(r);
                    opts.camera = cameras(c);
                    
                    tag = [current_class, fsep, iLab_idx2nameInstance(current_instance), ...
                            fsep, iLab_idx2nameCamera(opts.camera), fsep, iLab_idx2nameRotation(opts.rotation)];
               
                    idx = find(~cellfun('isempty', strfind(objInstCRnames, tag)));
                    region  = [x1(idx) y1(idx) x2(idx) y2(idx)];
                    x1_ = x1(idx); x2_ = x2(idx);
                    y1_ = y1(idx); y2_ = y2(idx);
                    

                    bImg = iLab_imgExist(opts);
                    if bImg == false
                        continue;
                    else
                        objCenter =  iLab_getCameraCenter(opts.camera+1);
                        im = iLab_readimg(opts);
                        impatch = im(y1_:y2_, x1_:x2_, :);
                        imwrite(impatch, fullfile(saveDir2, folderName, [tag '.jpg']));
                    end          
                end
            end
        end

    end
end


