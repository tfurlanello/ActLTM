% this script is used to compute the activations of the trained deCNN and
% AlexNet on test images of the iLab20M dataset


%% (1) from scratch
reps = [1 2 3 4];
nReps = numel(reps);

for e=1:nReps

scratchRoot = ['/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/fromscratch/exp1/', ...
                    'rep-', num2str(reps(e))];

deCNN_folderName    = 'iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000';
deCNN_folderName2   = 'iLab20M-iLab_arc_de_dagnn_2streams_woL2-w1.000-w1.000';
alexNet_folderName  = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

imdb_fileName = 'imdb.mat';
net_fileName  = 'net-epoch-10.mat';


imdb_files = { ...
            fullfile(scratchRoot, deCNN_folderName, imdb_fileName), ...
            fullfile(scratchRoot, deCNN_folderName2, imdb_fileName), ...
            fullfile(scratchRoot, alexNet_folderName, imdb_fileName)};
            

net_files = { ...
            fullfile(scratchRoot, deCNN_folderName, net_fileName), ...
            fullfile(scratchRoot, deCNN_folderName2, net_fileName), ...
            fullfile(scratchRoot, alexNet_folderName, net_fileName)};
        
        
saveDirs = { ...
            fullfile(scratchRoot, deCNN_folderName), ...
            fullfile(scratchRoot, deCNN_folderName2), ...
            fullfile(scratchRoot, alexNet_folderName)};

nNets = numel(net_files);

netInputsInfos       =   {...
                        {'imLeft', 'imRight', 'identity', 'transformation'}, ...
                        {'imLeft', 'imRight', 'identity', 'transformation'}, ...
                        {'imLeft', 'identity'}};
whichLayersToEvals   =  {...
                        {'dropoutlf1out'}, ...
                        {'dropoutlf1out'}, ...
                        {'dropoutlf1out', 'dropoutlf2out'}};


for n=1:nNets

    imdb_file = imdb_files{n};
    net_file  = net_files{n};
    saveDir   = saveDirs{n};
    netInputsInfo = netInputsInfos{n};
    whichLayersToEval = whichLayersToEvals{n};

    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %                                                       image database file
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    imdb = load(imdb_file);
    
    %---------------------------------------------------------------------
    % -------------------------------important: prune the duplicate images
    %---------------------------------------------------------------------
    btest = imdb.images.set == 3;
    idxTest = find(btest);
    testNames = imdb.images.name(1,btest);
    [uTestNames,idx_testNames,~] = unique(testNames);
    idxTestChosen = idxTest(idx_testNames);
    
    chosen_id = imdb.images.id(idxTestChosen);
    chosen_name = imdb.images.name(:,idxTestChosen);
    chosen_label = imdb.images.label(:,idxTestChosen);
    chosen_set = imdb.images.set(:,idxTestChosen);
    
    nonTest_id = imdb.images.id(~btest);
    nonTest_name = imdb.images.name(:,~btest);
    nonTest_label = imdb.images.label(:,~btest);
    nonTest_set = imdb.images.set(:,~btest); 
    
    imdb.images.id = cat(2, nonTest_id, chosen_id);
    imdb.images.name = cat(2, nonTest_name, chosen_name);
    imdb.images.set = cat(2, nonTest_set, chosen_set);
    imdb.images.label = cat(2, nonTest_label, chosen_label);
    %---------------------------------------------------------------------
    % -------------------------------important: prune the duplicate images
    %---------------------------------------------------------------------


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

%     whichLayersToEval   =  {};

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
end



%% (2) from AlexNet pretrained on iLab20M

for e=1:nReps

scratchRoot = ['/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-AlexNet/exp1/', ...
                    'rep-', num2str(reps(e))];

deCNN_folderName    = 'iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000';
deCNN_folderName2   = 'iLab20M-iLab_arc_de_dagnn_2streams_woL2-w1.000-w1.000';
alexNet_folderName  = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

imdb_fileName = 'imdb.mat';
net_fileName  = 'net-epoch-10.mat';


imdb_files = { ...
            fullfile(scratchRoot, deCNN_folderName, imdb_fileName), ...
            fullfile(scratchRoot, deCNN_folderName2, imdb_fileName), ...
            fullfile(scratchRoot, alexNet_folderName, imdb_fileName)};
            

net_files = { ...
            fullfile(scratchRoot, deCNN_folderName, net_fileName), ...
            fullfile(scratchRoot, deCNN_folderName2, net_fileName), ...
            fullfile(scratchRoot, alexNet_folderName, net_fileName)};
        
        
saveDirs = { ...
            fullfile(scratchRoot, deCNN_folderName), ...
            fullfile(scratchRoot, deCNN_folderName2), ...
            fullfile(scratchRoot, alexNet_folderName)};

nNets = numel(net_files);

netInputsInfos       =   {...
                        {'imLeft', 'imRight', 'identity', 'transformation'}, ...
                        {'imLeft', 'imRight', 'identity', 'transformation'}, ...
                        {'imLeft', 'identity'}};
whichLayersToEvals   =  {...
                        {'dropoutlf1out'}, ...
                        {'dropoutlf1out'}, ...
                        {'dropoutlf1out', 'dropoutlf2out'}};


for n=1:nNets

    imdb_file = imdb_files{n};
    net_file  = net_files{n};
    saveDir   = saveDirs{n};
    netInputsInfo = netInputsInfos{n};
    whichLayersToEval = whichLayersToEvals{n};

    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %                                                       image database file
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    imdb = load(imdb_file);
    
    %---------------------------------------------------------------------
    % -------------------------------important: prune the duplicate images
    %---------------------------------------------------------------------
    btest = imdb.images.set == 3;
    idxTest = find(btest);
    testNames = imdb.images.name(1,btest);
    [uTestNames,idx_testNames,~] = unique(testNames);
    idxTestChosen = idxTest(idx_testNames);
    
    chosen_id = imdb.images.id(idxTestChosen);
    chosen_name = imdb.images.name(:,idxTestChosen);
    chosen_label = imdb.images.label(:,idxTestChosen);
    chosen_set = imdb.images.set(:,idxTestChosen);
    
    nonTest_id = imdb.images.id(~btest);
    nonTest_name = imdb.images.name(:,~btest);
    nonTest_label = imdb.images.label(:,~btest);
    nonTest_set = imdb.images.set(:,~btest); 
    
    imdb.images.id = cat(2, nonTest_id, chosen_id);
    imdb.images.name = cat(2, nonTest_name, chosen_name);
    imdb.images.set = cat(2, nonTest_set, chosen_set);
    imdb.images.label = cat(2, nonTest_label, chosen_label);
    %---------------------------------------------------------------------
    % -------------------------------important: prune the duplicate images
    %---------------------------------------------------------------------


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

%     whichLayersToEval   =  {};

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
end


%% (3) from pretrained deCNN on iLab20M


for e=1:nReps

scratchRoot = ['/lab/igpu3/u/jiaping/washington-RGBD/results/ECCV/warmstart-iLab20M-deCNN/exp1/', ...
                    'rep-', num2str(reps(e))];

deCNN_folderName    = 'iLab20M-iLab_arc_de_dagnn_2streams_wL2-w0.050-w1.000-w1.000';
deCNN_folderName2   = 'iLab20M-iLab_arc_de_dagnn_2streams_woL2-w1.000-w1.000';
alexNet_folderName  = 'iLab20M-iLab_arc_de_dagnn_2streams_alexnet';

imdb_fileName = 'imdb.mat';
net_fileName  = 'net-epoch-10.mat';


imdb_files = { ...
            fullfile(scratchRoot, deCNN_folderName, imdb_fileName), ...
            fullfile(scratchRoot, deCNN_folderName2, imdb_fileName), ...
            fullfile(scratchRoot, alexNet_folderName, imdb_fileName)};
            

net_files = { ...
            fullfile(scratchRoot, deCNN_folderName, net_fileName), ...
            fullfile(scratchRoot, deCNN_folderName2, net_fileName), ...
            fullfile(scratchRoot, alexNet_folderName, net_fileName)};
        
        
saveDirs = { ...
            fullfile(scratchRoot, deCNN_folderName), ...
            fullfile(scratchRoot, deCNN_folderName2), ...
            fullfile(scratchRoot, alexNet_folderName)};

nNets = numel(net_files);

netInputsInfos       =   {...
                        {'imLeft', 'imRight', 'identity', 'transformation'}, ...
                        {'imLeft', 'imRight', 'identity', 'transformation'}, ...
                        {'imLeft', 'identity'}};
whichLayersToEvals   =  {...
                        {'dropoutlf1out'}, ...
                        {'dropoutlf1out'}, ...
                        {'dropoutlf1out', 'dropoutlf2out'}};


for n=1:nNets

    imdb_file = imdb_files{n};
    net_file  = net_files{n};
    saveDir   = saveDirs{n};
    netInputsInfo = netInputsInfos{n};
    whichLayersToEval = whichLayersToEvals{n};

    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %                                                       image database file
    %XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    imdb = load(imdb_file);
    
    %---------------------------------------------------------------------
    % -------------------------------important: prune the duplicate images
    %---------------------------------------------------------------------
    btest = imdb.images.set == 3;
    idxTest = find(btest);
    testNames = imdb.images.name(1,btest);
    [uTestNames,idx_testNames,~] = unique(testNames);
    idxTestChosen = idxTest(idx_testNames);
    
    chosen_id = imdb.images.id(idxTestChosen);
    chosen_name = imdb.images.name(:,idxTestChosen);
    chosen_label = imdb.images.label(:,idxTestChosen);
    chosen_set = imdb.images.set(:,idxTestChosen);
    
    nonTest_id = imdb.images.id(~btest);
    nonTest_name = imdb.images.name(:,~btest);
    nonTest_label = imdb.images.label(:,~btest);
    nonTest_set = imdb.images.set(:,~btest); 
    
    imdb.images.id = cat(2, nonTest_id, chosen_id);
    imdb.images.name = cat(2, nonTest_name, chosen_name);
    imdb.images.set = cat(2, nonTest_set, chosen_set);
    imdb.images.label = cat(2, nonTest_label, chosen_label);
    %---------------------------------------------------------------------
    % -------------------------------important: prune the duplicate images
    %---------------------------------------------------------------------


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

%     whichLayersToEval   =  {};

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
end









