% visualize the first layer features:

% (1) multiLevelInjection
load('/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera/iLab20M-alexnet-dagnn-multiLevelInjection-unstructured/net-epoch-8.mat');
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
multiLevelObjEnv = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
title('multi-Level Injection', 'fontsize', 30);

% (2) 2 label layers
load('/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera/iLab20M-alexnet-dagnn-catcam/net-epoch-10.mat');
conv1f = net.params(1).value;
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
labelsObjEnv = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
title('2 label layers: object + environment', 'fontsize', 30);

% (3) single object label layer
load('/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera/iLab20M-alexnet-simplenn-cat2/net-epoch-20.mat');
conv1f = net.layers{1}.weights{1};
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
labelsObj = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
title('1 label layer: object', 'fontsize', 30);
 

% (4) single environment layer
load('/lab/igpu3/projects/iLab20M-datasets-experiments/category-camera/iLab20M-alexnet-simplenn-cam/net-epoch-20.mat');
conv1f = net.layers{1}.weights{1};
D = reshape(conv1f, [size(conv1f,1)*size(conv1f,2)*size(conv1f,3) size(conv1f,4)]);
labelsEnv = plotCollage(D, [size(conv1f, 1) size(conv1f,2)]);
title('1 label layer: environment', 'fontsize', 30);


 


figure; 
subplot(221); imshow(multiLevelObjEnv); title('multi-Level Injection', 'fontsize', 30);
subplot(222); imshow(labelsObjEnv);     title('2 label layers: object + environment', 'fontsize', 30);
subplot(223); imshow(labelsObj);        title('1 label layer: object', 'fontsize', 30);
subplot(224); imshow(labelsEnv);        title('1 label layer: environment', 'fontsize', 30);