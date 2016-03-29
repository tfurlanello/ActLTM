% generate training and testing images
% now each image has as many as 10 repetitions in the training set

% case 1:
% for the same camera, there are two labels: two adjacent rotations, two
% skip rotations
% case 2:
% for the same rotation, there are two labels: two adjacent cameras, two
% skip cameras
% case 3:
% adjacent camera and adjacent rotations

saveDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/metadata-disentangling-rep5';
dataDir = '/lab/igpu3/u/jiaping/iLab20M-objRec/dataset/iLab20M-datasets-pilot-experiments/category-camera/metadata';

trainImgsInfoFile  =  'trainImagesInfo.mat';
testImgsInfoFile   =  'testImagesInfo.mat';

classes = iLab_getClasses;
mapTransform = iLab_de_genLabelSpace_f56;


mapNeiCam_s   =  iLab_de_getNeighboringCameras_sameRot;
mapNeiCam_d   =  iLab_de_getNeighboringCameras_diffRot;
mapNeiRot_s2  =  iLab_de_getNeighboringRotations_sameCam(2);
mapNeiRot_s1  =  iLab_de_getNeighboringRotations_sameCam(1);

separator = '-';


%========================================================================== 
%                                                           training images
%==========================================================================
    ftrain = fopen(fullfile(saveDir, 'train-candidate.txt'), 'w');

    load(fullfile(dataDir, trainImgsInfoFile));
    nTrain = numel(trainImagesInfo);
    names = fieldnames(trainImagesInfo);
    values = [];
    for f=1:numel(names)
        values = cat(1, values, cell2mat({trainImagesInfo.(names{f})}));
    end

    trainClasses = unique(classes(values(2,:)));
    mapObject = containers.Map(trainClasses, 1:numel(trainClasses));

    save(fullfile(saveDir, 'mappings.mat'), 'mapObject', 'mapTransform');

    
    
    tic
    for i=1:nTrain

        if rem(i, 1000) == 0
            i
            toc
        end

        ref_para     =  values(:,i);
        ref_camera   =  ref_para(4);
        ref_rotation =  ref_para(5);

        class = classes{ref_para(2)};
        ref_img_file = iLab_genImgFileName({'class', class, 'instance', ref_para(3), ...
            'background', ref_para(1), 'camera', ref_para(4), 'rotation', ref_para(5), ...
            'focus', ref_para(6), 'light', ref_para(7)});        

        %% case 1: the same rotation, neighboring cameras
        if mapNeiCam_s.isKey(ref_camera)
            neicams = mapNeiCam_s(ref_camera);
            for ii=1:numel(neicams)
                tar_para = ref_para;
                tar_para(4) = neicams(ii);

                tar_img_file = iLab_genImgFileName({'class', class, 'instance', tar_para(3), ...
                            'background', tar_para(1), 'camera', tar_para(4), 'rotation', tar_para(5), ...
                            'focus', tar_para(6), 'light', tar_para(7)});

                l_c1 = iLab_idx2nameCamera(ref_para(4));
                l_c2 = iLab_idx2nameCamera(tar_para(4));
                l_r  = iLab_idx2nameRotation(ref_para(5));

                transform = [l_c1 separator l_c2 separator l_r]; 

                transformIdx = mapTransform(transform);
                classIdx = mapObject(class);


                
                fprintf(ftrain, '%s %d %s %d %s %s\n', class, classIdx, transform, ...
                        transformIdx, ref_img_file, tar_img_file);



            end            
        end        

        %% case 2: the same camera, neighboring rotations
        if mapNeiRot_s2.isKey(ref_rotation)
           neirots = mapNeiRot_s2(ref_rotation);
           for ii=1:numel(neirots)
              tar_para = ref_para;
              tar_para(5) = neirots(ii);

                tar_img_file = iLab_genImgFileName({'class', class, 'instance', tar_para(3), ...
                            'background', tar_para(1), 'camera', tar_para(4), 'rotation', tar_para(5), ...
                            'focus', tar_para(6), 'light', tar_para(7)});

                l_c  = iLab_idx2nameCamera(ref_para(4));    
                l_r1  = iLab_idx2nameRotation(ref_para(5));
                l_r2  = iLab_idx2nameRotation(tar_para(5));

                transform = [l_c  separator l_r1 separator l_r2];   

                transformIdx = mapTransform(transform);
                classIdx = mapObject(class);

                
                fprintf(ftrain, '%s %d %s %d %s %s\n', class, classIdx, transform, ...
                        transformIdx, ref_img_file, tar_img_file);

           end
        end        

       %% case 3: adjacent cameras, but neighboring rotations
        if mapNeiRot_s1.isKey(ref_rotation)
            neirot = mapNeiRot_s1(ref_rotation);

            if mapNeiCam_d.isKey(ref_camera)
                neicams = mapNeiCam_d(ref_camera);

                for ii=1:numel(neicams)
                   tar_para = ref_para;
                   tar_para(5) = neirot;
                   tar_para(4) = neicams(ii);

                   tar_img_file = iLab_genImgFileName({'class', class, 'instance', tar_para(3), ...
                            'background', tar_para(1), 'camera', tar_para(4), 'rotation', tar_para(5), ...
                            'focus', tar_para(6), 'light', tar_para(7)});                   

                   l_c1 = iLab_idx2nameCamera(ref_para(4));
                   l_c2 = iLab_idx2nameCamera(tar_para(4));
                   l_r1 = iLab_idx2nameRotation(ref_para(5));
                   l_r2 = iLab_idx2nameRotation(tar_para(5));

                   transform = [l_c1  separator l_c2  separator  l_r1 separator l_r2];  

                   transformIdx = mapTransform(transform);
                   classIdx = mapObject(class);       

                
                    fprintf(ftrain, '%s %d %s %d %s %s\n', class, classIdx, transform, ...
                        transformIdx, ref_img_file, tar_img_file);


                end
            end
        end                
    end   

    fclose(ftrain);
    

%========================================================================== 
%                                                           test images
%==========================================================================    
    
   ftest = fopen(fullfile(saveDir, 'test-candidate.txt'), 'w');

    load(fullfile(dataDir, testImgsInfoFile));
    nTest = numel(testImagesInfo);
    names = fieldnames(testImagesInfo);
    values = [];
    for f=1:numel(names)
        values = cat(1, values, cell2mat({testImagesInfo.(names{f})}));
    end

    
    tic
    for i=1:nTest

        if rem(i, 1000) == 0
            i
            toc
        end

        ref_para     =  values(:,i);
        ref_camera   =  ref_para(4);
        ref_rotation =  ref_para(5);

        class = classes{ref_para(2)};
        ref_img_file = iLab_genImgFileName({'class', class, 'instance', ref_para(3), ...
            'background', ref_para(1), 'camera', ref_para(4), 'rotation', ref_para(5), ...
            'focus', ref_para(6), 'light', ref_para(7)});        

        %% case 1: the same rotation, neighboring cameras
        if mapNeiCam_s.isKey(ref_camera)
            neicams = mapNeiCam_s(ref_camera);
            for ii=1:numel(neicams)
                tar_para = ref_para;
                tar_para(4) = neicams(ii);

                tar_img_file = iLab_genImgFileName({'class', class, 'instance', tar_para(3), ...
                            'background', tar_para(1), 'camera', tar_para(4), 'rotation', tar_para(5), ...
                            'focus', tar_para(6), 'light', tar_para(7)});

                l_c1 = iLab_idx2nameCamera(ref_para(4));
                l_c2 = iLab_idx2nameCamera(tar_para(4));
                l_r  = iLab_idx2nameRotation(ref_para(5));

                transform = [l_c1 separator l_c2 separator l_r]; 

                transformIdx = mapTransform(transform);
                classIdx = mapObject(class);


                
                fprintf(ftest, '%s %d %s %d %s %s\n', class, classIdx, transform, ...
                        transformIdx, ref_img_file, tar_img_file);



            end            
        end        

        %% case 2: the same camera, neighboring rotations
        if mapNeiRot_s2.isKey(ref_rotation)
           neirots = mapNeiRot_s2(ref_rotation);
           for ii=1:numel(neirots)
              tar_para = ref_para;
              tar_para(5) = neirots(ii);

                tar_img_file = iLab_genImgFileName({'class', class, 'instance', tar_para(3), ...
                            'background', tar_para(1), 'camera', tar_para(4), 'rotation', tar_para(5), ...
                            'focus', tar_para(6), 'light', tar_para(7)});

                l_c  = iLab_idx2nameCamera(ref_para(4));    
                l_r1  = iLab_idx2nameRotation(ref_para(5));
                l_r2  = iLab_idx2nameRotation(tar_para(5));

                transform = [l_c  separator l_r1 separator l_r2];   

                transformIdx = mapTransform(transform);
                classIdx = mapObject(class);

                
                fprintf(ftest, '%s %d %s %d %s %s\n', class, classIdx, transform, ...
                        transformIdx, ref_img_file, tar_img_file);

           end
        end        

       %% case 3: adjacent cameras, but neighboring rotations
        if mapNeiRot_s1.isKey(ref_rotation)
            neirot = mapNeiRot_s1(ref_rotation);

            if mapNeiCam_d.isKey(ref_camera)
                neicams = mapNeiCam_d(ref_camera);

                for ii=1:numel(neicams)
                   tar_para = ref_para;
                   tar_para(5) = neirot;
                   tar_para(4) = neicams(ii);

                   tar_img_file = iLab_genImgFileName({'class', class, 'instance', tar_para(3), ...
                            'background', tar_para(1), 'camera', tar_para(4), 'rotation', tar_para(5), ...
                            'focus', tar_para(6), 'light', tar_para(7)});                   

                   l_c1 = iLab_idx2nameCamera(ref_para(4));
                   l_c2 = iLab_idx2nameCamera(tar_para(4));
                   l_r1 = iLab_idx2nameRotation(ref_para(5));
                   l_r2 = iLab_idx2nameRotation(tar_para(5));

                   transform = [l_c1  separator l_c2  separator  l_r1 separator l_r2];  

                   transformIdx = mapTransform(transform);
                   classIdx = mapObject(class);       

                
                    fprintf(ftest, '%s %d %s %d %s %s\n', class, classIdx, transform, ...
                        transformIdx, ref_img_file, tar_img_file);


                end
            end
        end                
    end   

    fclose(ftest);
