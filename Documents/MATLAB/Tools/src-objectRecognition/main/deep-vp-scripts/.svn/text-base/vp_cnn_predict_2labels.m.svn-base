%load('/lab/ilab/30/kai-vp/boundary_gen/vp-alexnet-simplenn-obj/net-epoch-1.mat');
% load('/lab/ilab/30/kai-vp/boundary_gen_extra/vp-alexnet-simplenn-obj/net-epoch-20.mat');
% imdb = load('/lab/ilab/30/kai-vp/boundary_gen/vp-alexnet-simplenn-obj/imdb.mat');

% load('/lab/ilab/30/kai-vp/boundary_gen_extra_noise/vp-alexnet-simplenn-obj/net-epoch-15.mat');

modelFile = '/home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/net-epoch-17.mat';
load(modelFile);
net = dagnn.DagNN.loadobj(net);

imdb = load('/home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/imdb.mat');

saveDir = '/home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2';

whoseLabel = {'object', 'environment'};
whichLayersToEval =  net.predictionsNames;

if ~exist(fullfile(saveDir, 'prediction.mat'), 'file')
    
    labels = iLab_cnn_predictBatch(net, imdb, [], 'dagnn', whoseLabel, whichLayersToEval);
    
    
%     labels_gt_simplenn = labels.groundtruth;
%     labels_pred_simplenn = labels.prediction;

    labels_gt_pitch = labels(1).groundtruth;
    labels_pred_pitch = labels(1).prediction;


    labels_gt_heading = labels(2).groundtruth;
    labels_pred_heading = labels(2).prediction;
    
    bTest = imdb.images.set == 3;
    testFiles = imdb.images.name(bTest);

    save(fullfile(saveDir, 'prediction.mat'), 'labels', 'labels_gt_simplenn', 'labels_pred_simplenn', ...
                    'labels_gt_pitch', 'labels_gt_heading', 'labels_pred_pitch', 'labels_pred_heading', ...
                    'testFiles');  
                
else
    
    load(fullfile(saveDir, 'prediction.mat'));
    
end


labels_gt_pitch = labels(1).groundtruth;
labels_pred_pitch = labels(1).prediction;


labels_gt_heading = labels(2).groundtruth;
labels_pred_heading = labels(2).prediction;


b_pitch = (labels_gt_pitch == labels_pred_pitch);
% | ...
%     labels_gt_pitch == labels_pred_pitch + 1 | ...
%     labels_gt_pitch == labels_pred_pitch - 1;
b_heading = labels_gt_heading == labels_pred_heading ;
% | ...
%     labels_gt_heading == labels_pred_heading + 1  | ...
%     labels_gt_heading == labels_pred_heading - 1;

b = b_pitch & b_heading;

acc_pitch = sum(b_pitch)/numel(b_pitch);
acc_heading = sum(b_heading)/numel(b_heading);
acc = sum(b)/numel(b);



nTest = numel(labels_gt_pitch);

 
gt_x = labels_gt_heading;
gt_y = labels_gt_pitch;

pred_x = labels_pred_heading;
pred_y = labels_pred_pitch;

pred_x_d = pred_x;
pred_y_d = pred_y;

% compute distances between gt and predictions
dist_gt2pred = zeros(nTest,1);
for j=1:nTest
    dist_gt2pred(j) = sqrt((gt_x(j) - pred_x_d(j)).^2 + ...
                    (gt_y(j) - pred_y_d(j)).^2 );
end


% 
% vp_display_vp_batch(imdb, labels_gt, labels_pred, map2x, map2y);
% 
% 
% % compute distance distributions
% 
% dists = zeros(nTest, 5);
% 
% for i=1:5
%     for j=1:nTest
%         dists(j,i) = sqrt((gt_x(j) - pred_x(j,i)).^2 + ...
%                         (gt_y(j) - pred_y(j,i)).^2 );
%     end
% end
% 
% for i=1:5
%     figure; 
%     hist(dists(:,i));
%     
% end
% 
