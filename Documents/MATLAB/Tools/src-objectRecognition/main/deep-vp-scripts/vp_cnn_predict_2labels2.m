%% ================================================================
modelFile = '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/net-epoch-17.mat';
load(modelFile);
net = dagnn.DagNN.loadobj(net);

imdb = load('/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/imdb.mat');
imageNames = imdb.images.name;
for i=1:numel(imageNames) 
    imageNames{i} = ['/lab/igpu3' imageNames{i}(7:end)]; 
end
imdb.images.name = imageNames;



saveDir = '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2';

whoseLabel = {'object', 'environment'};
whichLayersToEval =  net.predictionsNames;

if ~exist(fullfile(saveDir, 'prediction.mat'), 'file')
    
    labels  = vp_cnn_predictBatch(net, imdb, [], 'dagnn', whoseLabel, whichLayersToEval);
    
%     labels_gt_simplenn = labels.groundtruth;
%     labels_pred_simplenn = labels.prediction;

    labels_gt_pitch   = labels(1).groundtruth;
    labels_pred_pitch = labels(1).prediction;
    labels_prob_pitch = labels(1).probability;


    labels_gt_heading   = labels(2).groundtruth;
    labels_pred_heading = labels(2).prediction;
    labels_prob_heading = labels(2).probability;
    
    bTest = imdb.images.set == 3;
    testFiles = imdb.images.name(bTest);

    save(fullfile(saveDir, 'prediction.mat'), 'labels_gt_pitch', 'labels_gt_heading', ...
                                              'labels_pred_pitch', 'labels_pred_heading', ...
                                              'labels_prob_pitch', 'labels_prob_heading', ...
                                              'testFiles');  
                                          

    fid = fopen(fullfile(saveDir, 'prediction-pitch.txt'), 'w');        
    for i=1:numel(testFiles)
        fprintf(fid, '%d %d %d %d %d %d %.4f %.4f %.4f %.4f %.4f %s\n', labels_gt_pitch(i), ...
            labels_pred_pitch(i,1), labels_pred_pitch(i,2), labels_pred_pitch(i,3), labels_pred_pitch(i,4), labels_pred_pitch(i,5), ...
            labels_prob_pitch(i,1), labels_prob_pitch(i,2), labels_prob_pitch(i,3), labels_prob_pitch(i,4), labels_prob_pitch(i,5), ...
            testFiles{i});
    end
    fclose(fid);                                          
                
    
    fid = fopen(fullfile(saveDir, 'prediction-heading.txt'), 'w');        
    for i=1:numel(testFiles)
        fprintf(fid, '%d %d %d %d %d %d %.4f %.4f %.4f %.4f %.4f %s\n', labels_gt_heading(i), ...
            labels_pred_heading(i,1), labels_pred_heading(i,2), labels_pred_heading(i,3), labels_pred_heading(i,4), labels_pred_heading(i,5), ...
            labels_prob_pitch(i,1), labels_prob_heading(i,2), labels_prob_heading(i,3), labels_prob_heading(i,4), labels_prob_heading(i,5), ...
            testFiles{i});
    end
    fclose(fid);                                              
    
else
    
    load(fullfile(saveDir, 'prediction.mat'));
    
end

%% ================================================================
return;
