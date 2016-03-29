function [im,labels] = iLab_getBatchSimpleNN(imdb, batch, getBatch_opts, whoseLabel)
% inputs:
%           imdb  - image database information
%           batch - batch index
%           getBatch_opts - getbatch hyperparameters
%           whoseLabel - either 'object', or 'environment'

   getBatch_opts = iLab_nn_validateGetImageBatchParam(getBatch_opts);    
   images = strcat([imdb.imageDir filesep], imdb.images.name(1,batch)) ;
   im = iLab_getImageBatch(images, getBatch_opts) ;
%       im = vp_getImageBatchBinary(images, getBatch_opts) ;
        
    numAugments = getBatch_opts.numAugments; 
    
    if ~exist('whoseLabel', 'var') || isempty(whoseLabel)
        labels = [];
        return;
    end    
    
    if ischar(whoseLabel)
        whoseLabel = {whoseLabel};
    end
    assert(numel(whoseLabel) == 1);

     switch whoseLabel{1}                
        case 'object'
            label_obj = imdb.images.label(1,batch);
            label_obj = repmat(label_obj, [numAugments 1]);
            labels =  label_obj(:);
        case 'environment'
            label_env = imdb.images.label(2,batch);
            label_env = repmat(label_env, [numAugments 1]);
            labels =  label_env(:);
        otherwise
            error('un-recognized variables\n');
     end
     
    


end