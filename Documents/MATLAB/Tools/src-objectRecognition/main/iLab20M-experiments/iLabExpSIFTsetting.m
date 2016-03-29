% extract sift feature points from images with uniform background, and do
% kmeans clustering, and encode image with bag-of-visual words

% then use t-SNE to map the high-dimensional embedding to the 2D space, and
% visualize them


%% basic training and test information
objects = {'car', 'f1car', 'mil', 'tank', 'van'}; % now test only under 
                                                  % 5 object classes
                                                  
light               = 0;    % with all four lights on                                              
focus               = 1;    % now only use images with focus 1
uniform_backgrounds = 0;    % images with uniform backgrounds shouldn't be 
                            % in the training, but place them into test sets
                            
                            
% generate image lists with the above settings

%% (1) filter images
iLabInfo = load('../iLab20M-data-info/imagefiles-info.mat');

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
fBackgrounds = iLabInfo.backgroundsIdx == uniform_backgrounds;
% filter light
fLight = iLabInfo.lightsIdx == light;

bSelected = fClasses & fFocus & fBackgrounds & fLight;
 
 
%% (3) generate labels
classes         = iLabInfo.classes;
labelsSelected  = cell(numel(bSelected),1);

parfor i=1:length(bSelected)
    if bSelected(i) == false
        continue;
    end
    labelsSelected{i} = [classes{iLabInfo.objectsIdx(i)} '-' ...   
                            num2str(iLabInfo.instancesIdx(i))];
end
  
labelsSelected = labelsSelected(bSelected);
save('data-sift.mat', 'bSelected', 'labelsSelected');


