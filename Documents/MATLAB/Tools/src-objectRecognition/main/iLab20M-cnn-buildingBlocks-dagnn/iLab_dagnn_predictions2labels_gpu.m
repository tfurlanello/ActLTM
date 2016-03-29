function [pred_labels, probs] = ...
                iLab_dagnn_predictions2labels_gpu(predictionValues, topN, labelgraph)
% inputs:
%          predictionValues - predicted values by deep nets, should be gpu
%                             array
%          topN             - top N predictions
%          labelgraph       - used for structured prediction
% notes:   only support two kinds of loss function 'crossentropy' and
%          'softmax'
   if ~existsOnGPU(predictionValues)
       error('only support GPU predictions\n');
   end
   
   if ~exist('topN', 'var') || isempty(topN)
       topN = 1;
   end
   
   if ~exist('labelgraph', 'var') || isempty(labelgraph)
       labelgraph.isstructured  = false;
       labelgraph.labelgraph    = [];
   end
   
    opts.isstructured = false;
    opts.labelgraph   = [];
    labelgraph = vl_argparse(opts, labelgraph);
    
    if labelgraph.isstructured == false
      [X, predictionValuesIdx] = sort(predictionValues, 3, 'descend') ;
      pred_labels = squeeze(predictionValuesIdx(:,:,1:topN,:));      
	  
      % get the probabilities    
      % note: the max value should be subtracted first, to be computational
      % stable
      Xsize= size(X);
      Xmax = max(X, [], 3);      
      ex = exp(bsxfun(@minus, X, Xmax)) ;
      probs = ex ./ repmat(sum(ex,3), [1 1 Xsize(3) 1]);      
      probs = squeeze(probs(:,:, 1:topN, :));
      
    else    
      labelInteraction = labelgraph.labelgraph;
      X = predictionValues;
      inputSize   =  size(X);
      nCategories =  inputSize(3);
      Xmax = max(X,[],3) ;
      ex = exp(bsxfun(@minus, X, Xmax));
      pX = X - repmat(Xmax, [1 1 nCategories 1]) - ...
            repmat(log(sum(ex,3)), [1 1 nCategories 1]) ;                
      predictions = gpuArray(zeros(size(X), 'single'));
      for i=1:nCategories
          i_weight = reshape(labelInteraction(i,:), [1 1 nCategories 1]);
          i_weight = repmat(i_weight, [inputSize(1) inputSize(2) 1 inputSize(4)]);
          predictions(:,:,i,:) = sum(pX .* gpuArray(i_weight),3);
      end
      [X, chat] = max(predictions,[],3) ;
      pred_labels = squeeze(chat(:,:,1:topN,:));
        
      % compute the probabilities of the top N most possible labels
      probs = squeeze(X(:,:,1:topN,:));
      probs = exp(probs);
    
    end


end