% this script is used to compute the activations of alexNet
% (1) trained from scratch
% (2) trained from pretrained decnn
% (3) trained from pretrained alexnet


%% ============= pretrained deCNN ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

scratchRoot = '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/warmstart-iLab20M-deCNN-4096';
arcFolder = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

imdb_files = { ...
               fullfile(scratchRoot, 'n5', arcFolder, 'imdb.mat'), ...
               fullfile(scratchRoot, 'n10', arcFolder, 'imdb.mat'), ...
               fullfile(scratchRoot, 'n20', arcFolder, 'imdb.mat'), ...
               fullfile(scratchRoot, 'n40', arcFolder, 'imdb.mat')};
               

alexnet_net_files = { ...
               fullfile(scratchRoot, 'n5', arcFolder,  'net-epoch-35.mat'), ...
               fullfile(scratchRoot, 'n10', arcFolder, 'net-epoch-35.mat'), ...
               fullfile(scratchRoot, 'n20', arcFolder, 'net-epoch-35.mat'), ...
               fullfile(scratchRoot, 'n40', arcFolder, 'net-epoch-35.mat')};
    
alexnet_saveDirs = { ...
               fullfile(scratchRoot, 'n5', arcFolder), ...
               fullfile(scratchRoot, 'n10', arcFolder), ...
               fullfile(scratchRoot, 'n20', arcFolder), ...
               fullfile(scratchRoot, 'n40', arcFolder)};
           
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
	whichLayersToEval   =   {};
    
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


%% ============= scratch ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

scratchRoot = '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/fromscratch';
arcFolder = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

imdb_files = { ...
               fullfile(scratchRoot, 'n5', arcFolder, 'imdb.mat'), ...
               fullfile(scratchRoot, 'n10', arcFolder, 'imdb.mat'), ...
               fullfile(scratchRoot, 'n20', arcFolder, 'imdb.mat'), ...
               fullfile(scratchRoot, 'n40', arcFolder, 'imdb.mat')};
               

alexnet_net_files = { ...
               fullfile(scratchRoot, 'n5', arcFolder,  'net-epoch-35.mat'), ...
               fullfile(scratchRoot, 'n10', arcFolder, 'net-epoch-35.mat'), ...
               fullfile(scratchRoot, 'n20', arcFolder, 'net-epoch-35.mat'), ...
               fullfile(scratchRoot, 'n40', arcFolder, 'net-epoch-35.mat')};
    
alexnet_saveDirs = { ...
               fullfile(scratchRoot, 'n5', arcFolder), ...
               fullfile(scratchRoot, 'n10', arcFolder), ...
               fullfile(scratchRoot, 'n20', arcFolder), ...
               fullfile(scratchRoot, 'n40', arcFolder)};
           
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
	whichLayersToEval   =   {};
    
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



%% ============= pretrained alexnet ============================================
%% XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

scratchRoot = '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/warmstart-iLab20M-AlexNet-4096';
arcFolder = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

imdb_files = { ...
               fullfile(scratchRoot, 'n5', arcFolder, 'imdb.mat'), ...
               fullfile(scratchRoot, 'n10', arcFolder, 'imdb.mat'), ...
               fullfile(scratchRoot, 'n20', arcFolder, 'imdb.mat'), ...
               fullfile(scratchRoot, 'n40', arcFolder, 'imdb.mat')};
               

alexnet_net_files = { ...
               fullfile(scratchRoot, 'n5', arcFolder,  'net-epoch-35.mat'), ...
               fullfile(scratchRoot, 'n10', arcFolder, 'net-epoch-35.mat'), ...
               fullfile(scratchRoot, 'n20', arcFolder, 'net-epoch-35.mat'), ...
               fullfile(scratchRoot, 'n40', arcFolder, 'net-epoch-35.mat')};
    
alexnet_saveDirs = { ...
               fullfile(scratchRoot, 'n5', arcFolder), ...
               fullfile(scratchRoot, 'n10', arcFolder), ...
               fullfile(scratchRoot, 'n20', arcFolder), ...
               fullfile(scratchRoot, 'n40', arcFolder)};
           
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
	whichLayersToEval   =   {};
    
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



