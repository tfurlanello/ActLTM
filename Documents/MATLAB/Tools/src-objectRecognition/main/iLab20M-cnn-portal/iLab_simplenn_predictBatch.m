function [labels, imdb] = ...
            iLab_simplenn_predictBatch(net_cpu, imdb, subset, getBatch, opts_runTest)

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
            
opts = iLab_nn_validateRunTestParam(opts_runTest);
opts.numSubBatches = 1; %% much sure to fix this to be 1

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net = vl_simplenn_move(net_cpu, 'gpu') ;
else
  net = net_cpu ;
  net_cpu = [] ;
end
 
res = [] ;
labels_gt = [];
labels_pred = [];
for t=1:opts.batchSize:numel(subset)
  fprintf('batch %3d/%3d: \n',  ...
          fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  numDone = 0 ;
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart  = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd    = min(t+opts.batchSize-1, numel(subset)) ;
    batch       = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    [im, labels] = getBatch(imdb, batch) ;

    if opts.prefetch
      if s==opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      getBatch(imdb, nextBatch) ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

    % evaluate CNN
    net.layers{end}.class = labels ;
 	res = vl_simplenn(net, im, [], res, ...
                      'accumulate', s ~= 1, ...
                      'disableDropout', true, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn) ;

    % accumulate   errors
    
    predictions = gather(res(end-1).x) ;
    [~,predictions] = sort(predictions, 3, 'descend') ;
    pred = squeeze(predictions(:,:,1:5,:));
    
    labels_gt = cat(1,labels_gt, labels(:));
    labels_pred = cat(1, labels_pred, pred');    

    numDone = numDone + numel(batch) ;
  end


end

    labels = struct('groundtruth', {}, 'prediction', {});
    labels(1).groundtruth = labels_gt;
    labels(1).prediction = labels_pred;
    imdb.pLabels    =   labels_pred;
    imdb.gtLabels   =   labels_gt;

end