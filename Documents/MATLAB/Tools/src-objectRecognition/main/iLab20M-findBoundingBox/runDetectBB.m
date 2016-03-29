% detect bounding box
global pathBB;

categories = iLab_getClasses;
nCategories = numel(categories);
rng('shuffle');
rorder_class = randperm(nCategories);

saveDir         = pathBB.round3.doing;
error_lists     = textread(fullfile(pathBB.round1.outlies, 'lists.collage'), '%s\n');
img_blacklists  = textread(fullfile(pathBB.round1.outlies, 'lists.img'), '%s\n');

for c=1:nCategories
    
    cClass      = categories{rorder_class(c)};
    instances   = iLab_retrieveInstances({'class', cClass});
    opts.class  = cClass;
    rorder_instances = randperm(numel(instances));
    
    for i=1:numel(instances)
        ins = instances(rorder_instances(i));
        opts.instance = ins;
        
        bdoing = fullfile(saveDir, [opts.class '-instance-' num2str(opts.instance) '.doing']);
        if exist(bdoing, 'file')
            continue;
        end
        
        fid = fopen(bdoing, 'w');
        fprintf(fid, '1');
        fclose(fid);
        
%       1st round segmentation        
%         iLab_calObjBB(opts);
%       2nd rounding segmentation
        iLab_bb_calObjBB_2ndround(opts, error_lists, img_blacklists);
    end
    
end