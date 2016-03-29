%load('/lab/ilab/30/kai-vp/boundary_gen/vp-alexnet-simplenn-obj/net-epoch-1.mat');
% load('/lab/ilab/30/kai-vp/boundary_gen_extra/vp-alexnet-simplenn-obj/net-epoch-20.mat');
% imdb = load('/lab/ilab/30/kai-vp/boundary_gen/vp-alexnet-simplenn-obj/imdb.mat');

% load('/lab/ilab/30/kai-vp/boundary_gen_extra_noise/vp-alexnet-simplenn-obj/net-epoch-15.mat');

modelFile = '/lab/ilab/30/kai-vp/boundary_gen_extra/vp-alexnet-simplenn-obj/net-epoch-20.mat';
load(modelFile);

testDataDir = '/lab/igpu3/projects/deep_vp/equad_canny';
imdb = vp_cnn_setupdata('dataDir', testDataDir, 'lite', false) ;

saveDir = '/lab/ilab/30/kai-vp/boundary_gen_extra_noise';

whoseLabel = {'object'};

if ~exist(fullfile(saveDir, 'prediction.mat'), 'file')
    % saveDir = '/lab/ilab/30/kai-vp/boundary_gen/vp-alexnet-simplenn-obj';

    labels = iLab_cnn_predictBatch(net, imdb, [], 'simplenn', whoseLabel);
    labels_gt_simplenn = labels.groundtruth;
    labels_pred_simplenn = labels.prediction;

    save(fullfile(saveDir, 'prediction.mat'), 'labels_gt_simplenn', 'labels_pred_simplenn');     
else
    
    load(fullfile(saveDir, 'prediction.mat'));
    
end

nTest = numel(labels_gt_simplenn);

 
gt_x = mod(labels_gt_simplenn,13);
gt_y = floor(labels_gt_simplenn/13) + 1;

pred_x = mod(labels_pred_simplenn,13);
pred_y = floor(labels_pred_simplenn/13) + 1;

pred_x_d = mean(pred_x(:,1:3), 2);
pred_y_d = mean(pred_y(:,1:3), 2);

% compute distances between gt and predictions
dist_gt2pred = zeros(nTest,1);
for j=1:nTest
    dist_gt2pred(j) = sqrt((gt_x(j) - pred_x_d(j)).^2 + ...
                    (gt_y(j) - pred_y_d(j)).^2 );
end



vp_display_vp_batch(imdb, labels_gt, labels_pred, map2x, map2y);





 
% compute distance distributions

dists = zeros(nTest, 5);

for i=1:5
    for j=1:nTest
        dists(j,i) = sqrt((gt_x(j) - pred_x(j,i)).^2 + ...
                        (gt_y(j) - pred_y(j,i)).^2 );
    end
end

for i=1:5
    figure; 
    hist(dists(:,i));
    
end



