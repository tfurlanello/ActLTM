%% scripts to run deCNN (disentangling CNN) on iLab20M dataset

arc_types = {'iLab_arc_de_dagnn_2streams_woL2', ...
              'iLab_arc_de_dagnn_2streams_wL2', ...
              'iLab_arc_de_dagnn_2streams_alexnet'};
          
arc_current      = 3;          
bmemory          = true; 
balancingFactor  = 1;
networkInputs = { ...
                {'imLeft', 'imRight', 'identity', 'transformation'}, ...
                {'imLeft', 'imRight', 'identity', 'transformation'}, ...
                {'imLeft', 'identity'}};

import dagnn.*;
% number of images per class
nImgsPClass = [5 10 20 40];
nP           = numel(nImgsPClass);
dataDir = '/lab/igpu3/u/jiaping/imageNet2010/images'; 
rootTrainOrderDir = '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/training-orders';


rootSaveDirs  = { ...
                '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/fromscratch', ...
                '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/warmstart-iLab20M-AlexNet-4096', ...
                '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/warmstart-iLab20M-deCNN-4096'};
%             , ...
%                 '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/warmstart-iLab20M-AlexNet-1024', ...
%                 '/lab/igpu3/u/jiaping/imageNet2010/results/ECCV2016-nobnorm/warmstart-iLab20M-deCNN-1024'};

netInitParamFileName = 'net_init_param.mat';  
nExp = numel(rootSaveDirs);

nclasses_obj = 1000;
nclasses_transformation = 10;

nfactors_lf          =  2;
nfractions_lf        =  [7/8 1/8];
fBatchNormalization  =  false; % do not use batch normalization


for e=nExp:-1:1
    
    for p=1:nP
        
        bFactors = balancingFactor;
    %%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    %               setup hyperparameters for CNN                             %  
    %%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    % -------------------------------------------------------------------------
    %   parameters under this section are the only parameters needed to be
    %   modified, and simply keep other parameters fixed
    %   make sure: parameters are consistent
        param.modelType         = arc_types{arc_current};
        param.networkType       = 'dagnn';
        infoPortVars            = {'imLeft', 'imRight', 'identity', 'transformation'}; 
        train.derOutputs        = {'objectiveL2r',1, ...
                                        'objectiveObject', 1, 'objectiveTransformation', 1};


        bshare = true; % two streams of alexnet share the same convolutional parameters
        bshare_lf = true;

        trainOrderDir = fullfile(rootTrainOrderDir, ['n' num2str(nImgsPClass(p))]);
        if ~exist(trainOrderDir, 'dir')
            mkdir(trainOrderDir);
        end
        saveDir = fullfile(rootSaveDirs{e}, ['n' num2str(nImgsPClass(p))]);
        if ~exist(saveDir, 'dir')
            mkdir(saveDir);
        end
    % -------------------------------------------------------------------------
    %                                                              dataset info
    % -------------------------------------------------------------------------
        par_imdb.dataDir = dataDir;
        par_imdb.lite = false ;
     

        sfx = param.modelType ;
        if fBatchNormalization
            sfx = strcat(sfx, '-bnorm') ; 
        end
        if numel(bFactors) == 3
            expDir = fullfile(saveDir, sprintf('iLab20M-%s-w%.3f-w%.3f-w%.3f', sfx, ...
                                            bFactors(1), bFactors(2), bFactors(3))) ;
        elseif numel(bFactors) == 2
            expDir = fullfile(saveDir, sprintf('iLab20M-%s-w%.3f-w%.3f', sfx, ...
                                            bFactors(1), bFactors(2))) ;
        elseif numel(bFactors) == 1
            expDir = fullfile(saveDir, sprintf('iLab20M-%s', sfx));
        end

        if exist(expDir, 'dir')
            continue;
        else
             mkdir(expDir);
        end

        par_imdb.imdbFile = fullfile(expDir, 'imdb.mat');

        param.imdb = par_imdb;   
        param.expDir = expDir;
        param.trainOrderDir = trainOrderDir;

    % -------------------------------------------------------------------------
    %                                              architecture hyperparameters
    % -------------------------------------------------------------------------    
        arc.batchNormalization  = fBatchNormalization ;
        arc.conv.weightInitMethod = 'gaussian';
        arc.conv.scale          = 1.0;
        arc.conv.learningRate   = [1 2];
        arc.conv.weightDecay    = [1 0];
        arc.bnorm.learningRate  = [2 1];
        arc.bnorm.weightDecay   = [10 10];
        arc.norm.param          = [5 1 0.0001/5 0.75];
        arc.pooling.method      = 'max';
        arc.dropout.rate        = 0.5;
        arc.fc.size             = 4096;


        param.arc = arc;    
    % -------------------------------------------------------------------------
    %                                                 training hyperparameters
    % -------------------------------------------------------------------------    
        train.batchSize        = 128 ;
        train.numSubBatches    = 1 ;
        train.continue         = true ;
        train.gpus             = [1] ;
        train.prefetch         = false; %true ;
        train.sync             = false ;
        train.cudnn            = true ;
        train.expDir           = expDir ;
        train.orderDir         = trainOrderDir;


        if ~fBatchNormalization
          train.learningRate = logspace(-2, -4, 35) ; % 
        else
          train.learningRate = logspace(-1, -4, 15) ; % use batch normalization
        end 
        train.numEpochs = numel(train.learningRate) ;

        param.train = train;



    %%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    %                   run an architecture                                   %  
    %%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    

    % -------------------------------------------------------------------------
    %                                                   Database initialization
    % -------------------------------------------------------------------------
        if exist(par_imdb.imdbFile, 'file')
          imdb = load(par_imdb.imdbFile) ;
        else
          imdb = ImageNet_de_cnn_setupdata('dataDir', par_imdb.dataDir, 'lite', par_imdb.lite, 'n', nImgsPClass(p)) ;
          if ~exist(param.expDir, 'dir')
            mkdir(param.expDir) ;
          end
          save(par_imdb.imdbFile, '-struct', 'imdb') ;
        end


    % -------------------------------------------------------------------------
    %                 make sure different architectures run on the same order 
    % -------------------------------------------------------------------------
        nTrain =  numel(find(imdb.images.set==1));
        nepoch = 60;
        rng('shuffle');
        for ep=1:nepoch
            trainOrderFile = fullfile(param.trainOrderDir, sprintf('train-order-%d.mat', ep));
            if ~exist(trainOrderFile, 'file')
               trainOrder = randperm(nTrain); 
               save(trainOrderFile, 'trainOrder');
            end
        end     


    % -------------------------------------------------------------------------
    %               make sure different network architecture use the same
    %               initialized parameters
    % -------------------------------------------------------------------------
        if ~exist(fullfile(rootSaveDirs{e}, netInitParamFileName), 'file')
            net_init_param = iLab_arc_de_dagnn_2streams_wL2_rndw(nclasses_obj, nclasses_transformation,...
                                          bshare, bshare_lf, nfactors_lf, nfractions_lf, ...
                                          arc, struct('isstructured', false, 'labelgraph', [])); 

            net_init_param_ = net_init_param ;
            net_init_param = net_init_param_.saveobj() ;
            save(fullfile(rootSaveDirs{e}, netInitParamFileName), 'net_init_param') ;
        end
        if e==1 && exist(fullfile(rootSaveDirs{e}, netInitParamFileName), 'file')
            load(fullfile(rootSaveDirs{e}, netInitParamFileName));
            net_init_param = dagnn.DagNN.loadobj(net_init_param);
        end
        if e~=1 && exist(fullfile(rootSaveDirs{e}, netInitParamFileName), 'file')
            load(fullfile(rootSaveDirs{e}, netInitParamFileName));
            net_init_param = dagnn.DagNN.loadobj(net);
            clear net;
        end


    % -------------------------------------------------------------------------
    %                                                    Network initialization
    % -------------------------------------------------------------------------

        switch param.modelType

            case 'alexnet-dagnn-2streams-disentangling'

                net = iLab_arc_dagnn_2streams_disentangling(nclasses_obj, nclasses_transformation,...
                                                bshare, arc, struct('isstructured', false, 'labelgraph', [])); 

                train.derOutputs = {net.outputsNames{1}, balancingFactor(1), net.outputsNames{2}, balancingFactor(2), ...
                              net.outputsNames{3}, balancingFactor(3), net.outputsNames{4},balancingFactor(4)};

            case 'iLab_arc_de_dagnn_2streams_woL2'

                net = iLab_arc_de_dagnn_2streams_woL2(nclasses_obj, nclasses_transformation,...
                                                net_init_param, bshare, bshare_lf, nfactors_lf, nfractions_lf, ...
                                                 arc, struct('isstructured', false, 'labelgraph', [])); 

        %         train.derOutputs = {net.outputsNames{1}, balancingFactors(1), net.outputsNames{2}, balancingFactors(2), ...
        %                       net.outputsNames{3}, balancingFactors(3), net.outputsNames{4},balancingFactors(4)};

                train.derOutputs = {net.outputsNames{1}, bFactors(1), net.outputsNames{2}, bFactors(2)};


            case 'iLab_arc_de_dagnn_2streams_wL2'

                net = iLab_arc_de_dagnn_2streams_wL2(nclasses_obj, nclasses_transformation,...
                                                net_init_param, bshare, bshare_lf, nfactors_lf, nfractions_lf, ...
                                                arc, struct('isstructured', false, 'labelgraph', [])); 

                train.derOutputs = {net.outputsNames{1}, bFactors(1),...
                            net.outputsNames{2}, bFactors(2),  net.outputsNames{3}, bFactors(3)};


            case 'iLab_arc_de_dagnn_2streams_alexnet'
                net = iLab_arc_de_dagnn_2streams_alexnet(nclasses_obj, nclasses_transformation,...
                                                net_init_param, bshare, bshare_lf, nfactors_lf, nfractions_lf, ...
                                                arc, struct('isstructured', false, 'labelgraph', [])); 

                train.derOutputs = {net.outputsNames{1}, 1};



            otherwise
                error('un-recognizied model type\n');
        end

    %-------------------------------------------------------------------------
    %       compute image statistics (mean, RGB covariances etc)
    %-------------------------------------------------------------------------
        switch param.networkType
            case 'simplenn'
                getBatch_opts = net.normalization ;
            case 'dagnn'
                getBatch_opts = net.meta.normalization;

        %         net.meta.normalization.border = 256 - net.meta.normalization.imageSize(1:2) ;
                net.conserveMemory            = bmemory;
                infoPortVars                  = net.inputsNames;        

        end
        getBatch_opts.numThreads = 12 ;
        imageStatsPath = fullfile(expDir, 'imageStats.mat') ;
        if exist(imageStatsPath, 'file')
          load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
        else
          [averageImage, rgbMean, rgbCovariance] = ...
                iLab_getImageStatistics(imdb, getBatch_opts) ;
          save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
        end

        switch param.networkType
            case 'simplenn'        
                net.normalization.averageImage = rgbMean ;
            case 'dagnn'        
                net.meta.normalization.averageImage = rgbMean ;
        end

    % -------------------------------------------------------------------------
    %  everything is prepared, run  Stochastic gradient descent
    % -------------------------------------------------------------------------
        [v,d] = eig(rgbCovariance) ;
        getBatch_opts.transformation    = 'stretch' ;
        getBatch_opts.averageImage      = rgbMean ;
        getBatch_opts.rgbVariance       = 0.1*sqrt(d)*v' ;
        useGpu = numel(train.gpus) > 0 ;


        switch lower(param.networkType)
         case 'simplenn'
            fn = iLab_getBatchSimpleNNWrapper(getBatch_opts, whoseLabels) ;
            [net,info] = cnn_train(net, imdb, fn,  train, 'conserveMemory', true) ;
          case 'dagnn'
            fn = iLab_de_getBatchDagNNWrapper(getBatch_opts, useGpu, infoPortVars, networkInputs{arc_current}) ;
        %     train = rmfield(train, {'sync', 'cudnn'}) ;
            train = rmfield(train, 'sync') ;
            info = cnn_train_dag(net, imdb, fn, train) ;
        end

    

    end
end