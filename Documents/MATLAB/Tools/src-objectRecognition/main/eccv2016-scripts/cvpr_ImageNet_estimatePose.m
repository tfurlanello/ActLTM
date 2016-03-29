% this script is used to estimate pose of imageNet images used
% the pretrained model on iLab20M

% remember to get the probabilities (pose probabilities)


    imdb_file = '/lab/igpu3/u/jiaping/imageNet2010/results/CVPR-pose-estimation/imdb.mat';
    net_file  = '/lab/igpu3/u/jiaping/imageNet2010/results/CVPR-pose-estimation/net-pose.mat';
    saveDir   = '/lab/igpu3/u/jiaping/imageNet2010/results/CVPR-pose-estimation';

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
    net.inputsNames         = net.inputsNames(1:2);
    net.outputsNames = net.outputsNames(2);
    net.predictionsNames = net.predictionsNames(2);
    % make sure: evaluation only on the test set
    subsetTest  = find(imdb.images.set == 3);    
    subset = subsetTest;
    
    %% change the range of labels
    btest = imdb.images.set == 3;
    btestlabels = ones(2, sum(btest));
    imdb.images.label(:,btest) = btestlabels;
    
    
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
        [labelsProb, evalInfo] =  iLab_dagnn_deCNN_evalBatchPortal(net, imdb, subset, netInputsInfo, ...
                                                        whichLayersToEval, saveOpt); 

        save(maxEvalFile, 'labelsProb', 'evalInfo', '-v7.3');    
        clear evalInfo;
    end
    
    
    
    