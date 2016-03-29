
global pathBB;
categories = iLab_getClasses;

nCategories = numel(categories);
rng('shuffle');
rorder_class = randperm(nCategories);
rorder_class = 1:nCategories;

resDir  = pathBB.round1.res;
saveDir = pathBB.round1.outlies;

fimg = fopen(fullfile(saveDir, 'lists.img'), 'wt');
fcollage = fopen(fullfile(saveDir, 'lists.collage'), 'wt');

for c=1:nCategories
    
    cClass = categories{rorder_class(c)};
    
        
    instances = iLab_retrieveInstances({'class', cClass});
    opts.class = cClass;
    rorder_instances = randperm(numel(instances));
%     rorder_instances = 1:numel(instances);
    
    for i=1:numel(instances)
        ins = instances(rorder_instances(i));
        opts.instance = ins;
        
        fprintf(1, 'processing: %s, %d\n', opts.class, opts.instance);
       
        current_class = opts.class;
        current_instance = opts.instance;
% 
%         [outlieImgs, outlieCollage] = ...
%                 iLab_autoDetectOutlieSegmentation({'class', current_class, 'instance', current_instance});
%             

        [outlieImgs, outlieCollage] = ...
                iLab_bb_autoDiagnoseOutlieSegmentation({'class', current_class, 'instance', current_instance});            
            
        for k=1:numel(outlieImgs)
            fprintf(fimg, '%s\n', outlieImgs{k});
        end
        
        for k=1:numel(outlieCollage)
            fprintf(fcollage, '%s\n',outlieCollage{k});
            strcmd = ['cp' ' ' fullfile(resDir, outlieCollage{k}) ' ' saveDir];
            unix(strcmd);
        end

    end
end

fclose(fimg);
fclose(fcollage);