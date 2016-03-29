
modelFiles = {...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc1024/net-epoch-15.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc512/net-epoch-15.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc256/net-epoch-15.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc128/net-epoch-15.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc64/net-epoch-15.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc32/net-epoch-15.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc16/net-epoch-15.mat'};


imdbFiles = {...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc1024/imdb.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc512/imdb.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc256/imdb.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc128/imdb.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc64/imdb.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc32/imdb.mat', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc16/imdb.mat'};

   
saveDirs = {...
       '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc1024', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc512', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc256', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc128', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc64', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc32', ...
    '/lab/igpu3/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-obj-fc16'};


nModels = numel(modelFiles);

for m=nModels:-1:1
    
    fprintf(1, 'processing model: %d\n', m);
    modelFile = modelFiles{m};
	load(modelFile);
    net = dagnn.DagNN.loadobj(net);

    imdbFile = imdbFiles{m};
    imdb = load(imdbFile);

    imageNames = imdb.images.name;
    for i=1:numel(imageNames) 
        imageNames{i} = ['/lab/igpu3' imageNames{i}(7:end)]; 
    end
    imdb.images.name = imageNames;
    
    saveDir = saveDirs{m};

    whoseLabel = {'object'};
    whichLayersToEval =  net.predictionsNames;

    saveNameMat = 'prediction-prob.mat';
    saveNameTxt = 'prediction-prob.txt';
    
    if ~exist(fullfile(saveDir, saveNameMat), 'file')

        labels  = vp_cnn_predictBatch(net, imdb, [], 'dagnn', whoseLabel, whichLayersToEval);

        labels_gt   = labels(1).groundtruth;
        labels_pred = labels(1).prediction;
        probs       = labels(1).probability;

        bTest = imdb.images.set == 3;
        testFiles = imdb.images.name(bTest);

        save(fullfile(saveDir, saveNameMat), 'labels', 'labels_gt', 'labels_pred', 'probs', 'testFiles');  
        
        fid = fopen(fullfile(saveDir, saveNameTxt), 'w');        
        for i=1:numel(testFiles)
            fprintf(fid, '%d %d %d %d %d %d %.4f %.4f %.4f %.4f %.4f %s\n', labels_gt(i), ...
                labels_pred(i,1), labels_pred(i,2), labels_pred(i,3), labels_pred(i,4), labels_pred(i,5), ...
                probs(i,1), probs(i,2), probs(i,3), probs(i,4), probs(i,5), ...
                testFiles{i});
        end
        fclose(fid);
    else
        load(fullfile(saveDir, saveNameMat));
    end      
    
    
end

