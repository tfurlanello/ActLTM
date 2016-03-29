function preds = iLab_dagnn_MTL_predictSingle(net, im)
% current support both STL and MTL, i.e., net could be either a STL or MTL
% architecture. Note: currently don't support label graph
% input:
%           net     - trained dagnn architecture
%           im      - test image
% ouput: 
%           vargout - labels of the test image

    
    narginchk(2,2);
    if ~isa(net, 'dagnn.DagNN')
        error('Only support dagnn\n');
    end
    
    inputImgSize = net.meta.normalization.imageSize;
    inputH = inputImgSize(1);
    inputW = inputImgSize(2);
    nChannelsInput = inputImgSize(3);
    
    nChannelsIm = size(im,3);
    % convert to a color/grey image (input of the 'net')
    if nChannelsIm ~= nChannelsInput
        if nChannelsInput == 3 &&  nChannelsIm == 1
            im = cat(3, im, im, im) ;
        elseif nChannelsInput == 1 && nChannelsIm == 3
            im = rgb2gray(im);
        end
    end
    
    % resize
    ims = iLab_imrescale(im, [inputH inputW]);
    ims = single(ims);
    
    % normalization
    mes = net.meta.normalization.averageImage;
%     vars = net1.meta.normalization.rgbVariance;
%     vars = reshape(vars * randn(3,1),3,1);
%     mes = [mean(im(:,:,1)) mean(im(:,:,2)) mean(im(:,:,3))];
	
    vars = zeros(nChannelsInput,1);
    switch nChannelsInput
        case 3
            ims(:,:,1) = ims(:,:,1) - mes(1) - vars(1);
            ims(:,:,2) = ims(:,:,2) - mes(2) - vars(2);
            ims(:,:,3) = ims(:,:,3) - mes(3) - vars(3);
        case 1
            ims = ims - mes - vars;
    end

    
    imo = zeros(size(ims,1), size(ims,2), size(ims,3), 2);
    imo(:,:,:,1) = ims;
    imo(:,:,:,2) = ims;
    
    %============
    predictionsNames    =   net.predictionsNames;
    inputsNames  =  net.inputsNames;
    inputs = {inputsNames{1}, single(imo)};
    for i=2:numel(inputsNames)
        inputs = cat(2, inputs, inputsNames{i}, [1 1]);
    end
    
   
    net.conserveMemory            = 0;
    net.eval(inputs) ;
    
    nPred = numel(predictionsNames);
    preds = zeros(1, nPred);
    
    for i=1:nPred
        predictionValues = net.vars(net.getVarIndex(predictionsNames{i})).value ;
        predictions =  predictionValues;
        [~,predictions] = sort(predictions, 3, 'descend') ;
        pred = squeeze(predictions(:,:,1,:));         
        preds(i) = pred(1,1);
    end
    
end