

current_class = 'boat';
current_instance = 22;


medianbb = iLab_bb_getObjInstMaxBB({'class', current_class, 'instance', current_instance});
[objw,objh, objw_max, objh_max] =  ...
           iLab_bb_getObjInstScale({'class', current_class, 'instance', current_instance});
[objw_relax,objh_relax] = iLab_bb_getObjInstScaleRelax({'class', current_class, 'instance', current_instance});


% [outlieImgs, outlieCollage] = iLab_autoDetectOutlieSegmentation({'class', current_class, 'instance', current_instance});

cameras   = iLab_getCameras;
rotations = iLab_getRotations;
lights = iLab_getLights;


backgrounds     = iLab_retrieveBackgrounds({'class', current_class, 'instance', current_instance});
pureBackgrounds = [0 1 2 3 4 5 6];
realbackgrounds = setdiff(backgrounds, pureBackgrounds);


nBackgrounds =  numel(realbackgrounds);
nBackgroundsPure = numel(pureBackgrounds);
nCameras     =  numel(cameras);
nRotations   =  numel(rotations);
nLights      =  numel(lights);

opts.class = current_class;
opts.instance = current_instance;
opts.focus = 1;


%% use a rectangle to do the clipping
%{
crops_rec = [];
for b=2:2
%     opts.background = realbackgrounds(b);
	opts.background = pureBackgrounds(b);
	for c=1:nCameras
        for r=1:nRotations
            for l=1:nLights
                opts.light = lights(l);
                opts.rotation = rotations(r);
                opts.camera = cameras(c);
                bImg = iLab_imgExist(opts);
                if bImg == false
                    continue;
                else
                    im = iLab_readimg(opts);
                    xs = medianbb(1);
                    xe = medianbb(1) + medianbb(3);
                    ys = medianbb(2);
                    ye = medianbb(2) + medianbb(4);
                    crop = im(ys:ye, xs:xe, :);
                    crops_rec = [crops_rec crop(:)];
                end          
            end
        end
    end
 
end

if ~isempty(crops_rec)
    tmp = imCollage(crops_rec, [medianbb(4)+1 medianbb(3)+1]);
    figure; imshow(tmp);
end  
%}
%% use a square to do the clipping
[objw,objh, objw_max, objh_max]
crops_square = [];
objScale = max([objw_max objh_max]);
objScale = max([objw objh]);
objScale = max([objw_relax objh_relax]);

tmpScale1 = max([objw_max objh_max]);
tmpScale2 = max([objw_relax objh_relax]);
objScale = tmpScale1 + round(3*(tmpScale2-tmpScale1)/5);

for b=2:2
%     opts.background = realbackgrounds(b);
	opts.background = pureBackgrounds(b);
	for c=1:nCameras
        for r=1:nRotations
            for l=1:nLights
                opts.light = lights(l);
                opts.rotation = rotations(r);
                opts.camera = cameras(c);
                bImg = iLab_imgExist(opts);
                if bImg == false
                    continue;
                else
                    objCenter =  iLab_getCameraCenter(opts.camera+1);
                    im = iLab_readimg(opts);
                    crop = iLab_bb_intelligentClipper(im, objScale, objCenter);
                    crops_square = [crops_square crop(:)];
                end          
            end
        end
    end
 
end

if ~isempty(crops_square)
    tmp = imCollage(crops_square, [objScale objScale]);
    figure; imshow(tmp);
end 

