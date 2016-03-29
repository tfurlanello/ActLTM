% training with the aid of camera parameters
% generate training and test data

% training and testing images:
%                       - light = 0 (all lights on)
%                       - focus = 1
%                       - backgrounds ~= 0:6


%% basic training and test information
objects             = {'car', 'f1car', 'heli', 'plane', 'pickup', ...
                              'mil', 'monster', 'semi', 'tank', 'van'};  % 10 categories
focus               = 1;    % now only use images with focus 1
uniform_backgrounds = 0:6;  % using real backgrounds
light               = 0;    % currently, with no directional light (4 lights on)
 

% generate training image lists & labels
%          test image lists & labels
% note: there are 2 kinds of labels ( category and camera)

%% (1) filter images
iLabInfo    = iLab_getImgParameters;
nImages     = length(iLabInfo.objectsIdx);

% filter focus
fFocus = iLabInfo.focusIdx == focus;
% filter light
fLight = iLabInfo.lightsIdx == light;
% filter backgrouds
fBackgrounds = zeros(nImages,1) > 1.0;
for i=1:length(uniform_backgrounds)
    fBackgrounds = fBackgrounds | (iLabInfo.backgroundsIdx == uniform_backgrounds(i));
end
fBackgrounds = ~ fBackgrounds;
fCandidate = fFocus & fLight & fBackgrounds;
bTrain = zeros(nImages,1) > 1.0;
bTest  = zeros(nImages,1) > 1.0;

ratio_test = 1/4;

% filter classes
rng('shuffle');
fClasses = zeros(nImages,1) > 1.0;
for o=1:length(objects)
    o_object = objects{o};
    o_flag     = iLab_getClassIdx(o_object) == iLabInfo.objectsIdx;
    
    i_idx    =  iLabInfo.instancesIdx(o_flag);
    u_i_idx  =  sort(unique(i_idx), 'ascend');
    rorder   =  randperm(numel(u_i_idx));
    u_i_idx  = u_i_idx(rorder);
    i_flag_test = zeros(nImages,1) > 1.0;
    
    for i=1:round(ratio_test*numel(u_i_idx))
        i_flag_test = i_flag_test | (iLabInfo.instancesIdx == u_i_idx(i));
    end
    
    i_flag_train = ~i_flag_test;
    
    f_o_train = o_flag & i_flag_train;
    f_o_test = o_flag & i_flag_test;
    
    bTrain = bTrain | f_o_train;
    bTest = bTest | f_o_test;
    
    
end

bTrain = bTrain & fCandidate;
bTest  = bTest & fCandidate;
 

%% (3) generate labels
classes        = iLabInfo.classes;
labelsCatTrain = cell(nImages,1);
labelsCatTest  = cell(nImages,1);
labelsCamTrain = cell(nImages,1);
labelsCamTest  = cell(nImages,1);

% labelsTrain  = strcat(classes(iLabInfo.objectsIdx(bTrain)), ...
%              num2str(iLabInfo.instancesIdx(bTrain)));

parfor i=1:length(bTrain)
    if bTrain(i) == false
        continue;
    end
    labelsCatTrain{i} = classes{iLabInfo.objectsIdx(i)};
    labelsCamTrain{i} = iLab_genLabelCR({'camera', iLabInfo.camerasIdx(i), ...
                                            'rotation', iLabInfo.rotationsIdx(i)});
end

parfor i=1:length(bTest)
    if bTest(i) == false
        continue;
    end
    labelsCatTest{i} = classes{iLabInfo.objectsIdx(i)};
    labelsCamTest{i} = iLab_genLabelCR({'camera', iLabInfo.camerasIdx(i), ...
                    'rotation', iLabInfo.rotationsIdx(i)});
end


save('data-category-camera.mat', 'bTrain', 'bTest', ...
            'labelsCatTrain', 'labelsCatTest', ...
            'labelsCamTrain', 'labelsCamTest');


