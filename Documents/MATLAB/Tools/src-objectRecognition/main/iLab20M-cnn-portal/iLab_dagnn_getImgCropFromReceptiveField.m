function imdb = iLab_dagnn_getImgCropFromReceptiveField(net, imdb, subset, ...
                                                    whichLayerWhichChannel, whoseLabels)
% get a cropped image patch, which has the largest firing at the predefined
% layer

%% inputs:
%           net     - a trained dagnn model
%           imdb    - image database 
%           subset  - the index of images to be used
%           whichLayerWhichChannel - a struct specifying which channel,
%                                    which layer
%           getBatch - wrapper of image batch reader
%           opts_runBatch - options to run a batch of images


% this function is specifically designed for prediction of two label layers            
% -------------------------------------------------------------------------
    %{
    opts.batchSize      =   128 ;
    opts.numSubBatches  =   1 ;
    opts.gpus           =   1 ; % which GPU devices to use (none, one, or more)
    opts.conserveMemory =   false ;
    opts.sync           =   false ;
    opts.prefetch       =   false ;
    opts.cudnn          =   true ;

    opts = vl_argparse(opts, opts_in) ; 
    %}

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                    parameters to run test
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    nn_opts_runBatch.batchSize       = 128 ;
    nn_opts_runBatch.numSubBatches   = 1 ;
    nn_opts_runBatch.prefetch        = false ;
    nn_opts_runBatch.gpus            = 1 ;
    nn_opts_runBatch.sync            = false ;
    nn_opts_runBatch.cudnn           = true ;
    nn_opts_runBatch.conserveMemory  = false ;
    opts = iLab_nn_validateRunTestParam(nn_opts_runBatch); 
    opts.numSubBatches = 1; %% much sure to fix this to be 1 
    opts.gpus = 1; %% use gpu to do evaluations

    
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                 fetch image batch
%                                                 parameters
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    dagnn_opts_getbatch.numThreads      =   12;
    dagnn_opts_getbatch.useGpu          =   true;
    dagnn_opts_getbatch.transformation  =   'stretch' ;
    dagnn_opts_getbatch.numAugments     =   1;
    dagnn_opts_getbatch = iLab_nn_validateGetImageBatchParam(dagnn_opts_getbatch);
    
    useGpu = 1;
    opts_getbatch   = vl_argparse(dagnn_opts_getbatch, net.meta.normalization);
    getBatch        = iLab_getBatchDagNNWrapper(opts_getbatch, useGpu, net.inputsNames, ...
                                                    whoseLabels);

%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                     which layer, which channel to
%                                     evaluate
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    layerInfo.layer         =   'pool5';
    layerInfo.channel       =   1;
    whichLayerWhichChannel  =   vl_argparse(layerInfo, whichLayerWhichChannel);
    
    evalLayer   = whichLayerWhichChannel.layer;
    evalChannel = whichLayerWhichChannel.channel;
    

    numGpus = numel(opts.gpus) ;
    if numGpus >= 1
      net.move('gpu') ;
    end
    
    
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
%                                                       do batch evaluation
%XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   
    rfs = net.getVarReceptiveFields(net.vars(1).name);
    rfs = rfs(net.getVarIndex(evalLayer));
    
    inputImgSize = net.meta.normalization.imageSize;
    
    num = 0 ;      
    stimuli = struct('image', {},...
                     'patch', {});
    cntImg = 0;

     for t=1:opts.batchSize:numel(subset)

          for s=1:opts.numSubBatches
            % get this image batch and prefetch the next
            batchStart = t + (labindex-1) + (s-1) * numlabs ;
            batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
            batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            num = num + numel(batch) ;
            if numel(batch) == 0, continue ; end

            inputs = getBatch(imdb, batch) ;
            
            % read raw image
            images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
            rawImgs = vl_imreadjpeg(images, 'numThreads', 12) ;

            if opts.prefetch
              if s == opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
              else
                batchStart = batchStart + numlabs ;
              end
              nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
              getBatch(imdb, nextBatch) ;
            end
              net.eval(inputs) ;
          end         
        
       %%  get evaluation values at channel k of layer n
        values = net.vars(net.getVarIndex(evalLayer)).value ;
        layerSize = size(values);
        values = gather(values(:,:,evalChannel,:));
        
        values  = reshape(values, [layerSize(1)*layerSize(2) layerSize(4)]);
        [~,ind]      = max(values,[],1);
        [indy, indx] = ind2sub([layerSize(1) layerSize(2)], ind);
        
        for n=1:layerSize(4)
            y = indy(n);
            x = indx(n);
            
            cy = (y-1)*rfs.stride(1) + rfs.offset(1);
            cx = (x-1)*rfs.stride(2) + rfs.offset(2);
            
            dy1 = max(1, cy - floor(rfs.size(1)/2));
            dy2 = min(inputImgSize(1), cy + floor(rfs.size(1)/2));
            dy = dy1:dy2;
            
            dx1 = max(1, cx - floor(rfs.size(2)/2));
            dx2 = min(inputImgSize(2), cx + floor(rfs.size(2)/2));  
            dx = dx1:dx2;
            
            cntImg = cntImg + 1;
            
            rawImg = rawImgs{n};
            rawImg = imresize(rawImg, [inputImgSize(1) inputImgSize(2)]);
            stimuli(cntImg).image = rawImg;
            stimuli(cntImg).patch = imresize(rawImg(dy, dx, :), rfs.size);
        end
        
     end
     
     
 
     imdb.stimuli = struct( 'subset',    {}, ...
                            'layer',    {}, ...
                            'channel',  {}, ...
                            'crop',     {});
     imdb.stimuli(1).layer              = evalLayer;
     imdb.stimuli(1).channel            = evalChannel;
     imdb.stimuli(1).subset             = subset;
     imdb.stimuli(1).crop               = stimuli;

     
     net.reset() ;
     net.move('cpu') ;


end