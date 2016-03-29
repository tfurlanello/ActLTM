% lights invariance
% generate training and test data

% training images: with individual light on
% test images: with all lights on


%% basic training and test information
objects = {'car', 'f1car', 'mil', 'tank', 'van'}; % now test only under 
                                                  % 5 object classes
focus               = 1;    % now only use images with focus 1
uniform_backgrounds = 0:6;  % images with uniform backgrounds shouldn't be 
                            % in the training, but place them into test sets
                            
lights_train    = 1:4;      % training, individual lights on
lights_test     = 0;        % testing, all 4 lights on
                            
% generate training image lists & labels
%          test image lists & labels
% note: label is instance level, instead of object-level

%% (1) filter images
iLabInfo = load('../iLab20M-data-info/imagefiles-info.mat');
% fid         = fopen('../iLab20M-data-info/imagefiles-lists.txt', 'r');
% imageNames  = textscan(fid, '%s\n');
% nImages     = length(imageNames);
% fclose(fid);
nImages     = length(iLabInfo.objectsIdx);
% filter classes
fClasses = zeros(nImages,1) > 1.0;
for i=1:length(objects)
    i_object = objects{i};
    flag     = getClassIdxiLab(i_object) == iLabInfo.objectsIdx;
    fClasses = fClasses | flag;
end
% filter focus
fFocus = iLabInfo.focusIdx == focus;
% filter backgrouds
fBackgrounds = zeros(nImages,1) > 1.0;
for i=1:length(uniform_backgrounds)
    fBackgrounds = fBackgrounds | (iLabInfo.backgroundsIdx == uniform_backgrounds(i));
end
fBackgrounds = ~ fBackgrounds;

flagSelected = fClasses & fFocus & fBackgrounds;

%% (2) prepare training and test
% train: with individual light on
% test:  with all lights on

fLightsTrain = zeros(nImages, 1) > 1.0;
fLightsTest = zeros(nImages,1) > 1.0;

for i=1:length(lights_train)
    fLightsTrain = fLightsTrain | iLabInfo.lightsIdx == lights_train(i);
end

for i=1:length(lights_test)
    fLightsTest = fLightsTest | iLabInfo.lightsIdx == lights_test(i);
end

bTrain = flagSelected & fLightsTrain;
bTest  = flagSelected & fLightsTest;

%% (3) generate labels
classes = iLabInfo.classes;
labelsTrain = cell(sum(bTrain),1);
labelsTest  = cell(sum(bTest),1);

% labelsTrain  = strcat(classes(iLabInfo.objectsIdx(bTrain)), ...
%              num2str(iLabInfo.instancesIdx(bTrain)));

parfor i=1:length(bTrain)
    if bTrain(i) == false
        continue;
    end
    labelsTrain{i} = [classes{iLabInfo.objectsIdx(i)} '-' ...   
                            num2str(iLabInfo.instancesIdx(i))];
end

parfor i=1:length(bTest)
    if bTest(i) == false
        continue;
    end
    labelsTest{i} = [classes{iLabInfo.objectsIdx(i)} '-' ...   
                            num2str(iLabInfo.instancesIdx(i))];
end

u_labelsTrain = unique(labelsTrain(bTrain));
u_labelsTest = unique(labelsTest(bTest));

save('data-lightsInvariance.mat', 'bTrain', 'bTest', ...
            'labelsTrain', 'labelsTest', 'u_labelsTrain', 'u_labelsTest');








