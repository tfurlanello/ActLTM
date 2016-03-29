% this script is used to compute the activations of the trained deCNN and
% AlexNet on test images of the iLab20M dataset

imdb_files = { ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb/imdb-f7.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb/imdb-f11.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb/imdb-f18.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/test-imdb/imdb-f56.mat'};

%% ============= deCNN ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

deCNN_net_files = { ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f7/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000/net-epoch-6.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f11/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000/net-epoch-6.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f18/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000/net-epoch-4.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f56/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000/net-epoch-6.mat'};
    
deCNN_saveDirs = { ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f7/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f11/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f18/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f56/iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000'};

nNets = numel(deCNN_net_files);


for n=1:nNets

    imdb_file = imdb_files{n};
    net_file  = deCNN_net_files{n};
    saveDir   = deCNN_saveDirs{n};

    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %                                                       image database file
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    imdb = load(imdb_file);

    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %%                                            a trained dagnn model
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    net = load(net_file);
    net = net.net;
    net = dagnn.DagNN.loadobj(net);
    net.conserveMemory      =  false;
    net.accumulateParamDers =  false;    
    % make sure: evaluation only on the test set
    subsetTest  = find(imdb.images.set == 3);    
    subset = subsetTest;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %-------------------------------- first save intermediate evaluation values 
    netInputsInfo       =   {'imLeft', 'imRight', 'identity', 'transformation'};
    if  n==4 || n==3
        whichLayersToEval   =  {'dropoutlf1out'};
    else
        whichLayersToEval   =  {};
    end
    % raw activation
    saveOpt             = 'max';   
    evalFileName        = 'test-evalInfo.mat';   
    maxSaveDir = fullfile(saveDir, [saveOpt 'Activation']);
    if ~exist(maxSaveDir, 'dir')
        mkdir(maxSaveDir);
    end
    maxEvalFile = fullfile(maxSaveDir, evalFileName);
    if ~exist(maxEvalFile, 'file')
        [~, evalInfo] =  iLab_dagnn_deCNN_evalBatchPortal(net, imdb, subset, netInputsInfo, ...
                                                        whichLayersToEval, saveOpt); 

        save(maxEvalFile, '-struct', 'evalInfo', '-v7.3');    
        clear evalInfo;
    end


end



%% ============= AlexNet ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


alexnet_net_files = { ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f7/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/net-epoch-6.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f11/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/net-epoch-6.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f18/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/net-epoch-4.mat', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f56/iLab20M-iLab_arc_de_dagnn_2streams_alexnet/net-epoch-6.mat'};
    
alexnet_saveDirs = { ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f7/iLab20M-iLab_arc_de_dagnn_2streams_alexnet', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f11/iLab20M-iLab_arc_de_dagnn_2streams_alexnet', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f18/iLab20M-iLab_arc_de_dagnn_2streams_alexnet', ...
    '/lab/igpu3/u/jiaping/iLab20M-objRec/CNN-results/ECCV2016/exp2/f56/iLab20M-iLab_arc_de_dagnn_2streams_alexnet'};

nNets = numel(alexnet_net_files);


for n=1:nNets

    imdb_file = imdb_files{n};
    net_file  = alexnet_net_files{n};
    saveDir   = alexnet_saveDirs{n};

    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %                                                       image database file
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    imdb = load(imdb_file);

    if ~exist(saveDir, 'dir')
        mkdir(saveDir);
    end

    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %%                                            a trained dagnn model
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    net = load(net_file);
    net = net.net;
    net = dagnn.DagNN.loadobj(net);
    net.conserveMemory      =  false;
    net.accumulateParamDers =  false;    
    % make sure: evaluation only on the test set
    subsetTest  = find(imdb.images.set == 3);    
    subset = subsetTest;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %-------------------------------- first save intermediate evaluation values 
    netInputsInfo       =   {'imLeft', 'identity'};
    
    if n==4 || n==3
        whichLayersToEval   =  {'dropoutlf1out', 'dropoutlf2out'};
    else
        whichLayersToEval   =  {};
    end
    % raw activation
    saveOpt             = 'max';   
    evalFileName        = 'test-evalInfo.mat';   
    maxSaveDir = fullfile(saveDir, [saveOpt 'Activation']);
    if ~exist(maxSaveDir, 'dir')
        mkdir(maxSaveDir);
    end
    maxEvalFile = fullfile(maxSaveDir, evalFileName);
    if ~exist(maxEvalFile, 'file')
        [~, evalInfo] =  iLab_dagnn_deCNN_evalBatchPortal(net, imdb, subset, netInputsInfo, ...
                                                        whichLayersToEval, saveOpt); 

        save(maxEvalFile, '-struct', 'evalInfo', '-v7.3');    
        clear evalInfo;
    end


end





