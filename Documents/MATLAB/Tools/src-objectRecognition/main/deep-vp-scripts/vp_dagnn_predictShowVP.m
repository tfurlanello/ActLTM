function pred = vp_dagnn_predictShowVP(net, im)
    
    if ~exist('net','var') || isempty(net)    
%         load('/lab/ilab/30/kai-vp/google_dataset-grey/vp-alexnet-dagnn-obj/net-epoch-18.mat');
        load('/home2/u/kai/deep_vp/results/google_dataset-grey-400K/vp-alexnet-dagnn-obj/net-epoch-20.mat');        
%         load('/home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/net-epoch-17.mat');
        net = dagnn.DagNN.loadobj(net);
    end

    % convert to grey image
    imc = im;
    im = rgb2gray(im);
    
    im = repmat(im, [1 1 3]);
    ims = iLab_imrescale(im, [227 227]);
    imc = iLab_imrescale(imc, [227 227]);
    ims = single(ims);
    
    mes = net.meta.normalization.averageImage;
%     vars = net.meta.normalization.rgbVariance;
%     vars = reshape(vars * randn(3,1),3,1);
%     mes = [mean(im(:,:,1)) mean(im(:,:,2)) mean(im(:,:,3))];
    vars = zeros(3,1);
    ims(:,:,1) = ims(:,:,1) - mes(1) - vars(1);
    ims(:,:,2) = ims(:,:,2) - mes(2) - vars(1);
    ims(:,:,3) = ims(:,:,3) - mes(3) - vars(1);

     
    imo = zeros(size(ims,1), size(ims,2), size(ims,3), 2);
    imo(:,:,:,1) = ims;
    imo(:,:,:,2) = ims;
    
    %============
    predictionsNames    =   net.predictionsNames;


    inputsNames  =  net.inputsNames;
%     inputs.(inputsNames{1}) = imo;
%     inputs.(inputsNames{2}) = [1 1];
    
    inputs = {inputsNames{1}, single(imo), inputsNames{2}, [1 1]};
%     inputs = {inputsNames{1}, single(imo), inputsNames{2}, [1 1], inputsNames{3}, [1 1]};

    net.conserveMemory            = 0;
    net.eval(inputs) ;
    predictionValues = net.vars(net.getVarIndex(predictionsNames)).value ;
    
    predictions =  predictionValues;
%     probPred =     

    [~,predictions] = sort(predictions, 3, 'descend') ;
    pred = squeeze(predictions(:,:,1:5,:));         
    
    pred = pred(:,1);
    
    
    load('/lab/ilab/30/kai-vp/google_dataset/vp-alexnet-simplenn-obj/mapping.mat');
    mapy = containers.Map(mapping(:,1),mapping(:,3));
    mapx = containers.Map(mapping(:,1),mapping(:,2));
    
    
%     figure; 
    imshow(uint8(iLab_imrescale(imc, [300 300])));
    hold on;
    
 
    colors = {'red', 'green', 'blue'};
    for i=1:3
        x = mapx(pred(i));
        y = mapy(pred(i));
        scatter(x, y ,80, 'MarkerEdgeColor', colors{i}, 'linewidth', 5);
        hold on;
    end
    

end