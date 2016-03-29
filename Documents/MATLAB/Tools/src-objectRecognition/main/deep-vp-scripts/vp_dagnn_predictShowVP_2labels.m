function vp_dagnn_predictShowVP_2labels(net, im) %b_conf = 
    
    if ~exist('net','var') || isempty(net)    
%         load('/lab/ilab/30/kai-vp/google_dataset-grey/vp-alexnet-dagnn-obj/net-epoch-18.mat');
        load('/home2/u/kai/deep_vp/results/google_dataset-grey-1M/vp-alexnet-dagnn-multiLevelInjection-fc2/net-epoch-17.mat');
        net = dagnn.DagNN.loadobj(net);
%         net.move('cpu');
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
    
    inputs = {inputsNames{1}, single(imo), inputsNames{2}, [1 1], inputsNames{3}, [1 1]};

    net.conserveMemory            = 0;
    net.eval(inputs) ;
    
    pitchName = predictionsNames{1};
    headingName = predictionsNames{2};
    
    predictionValues_pitch = net.vars(net.getVarIndex(pitchName)).value ;
    predictionValues_heading = net.vars(net.getVarIndex(headingName)).value ;
    
%     probPred =     

    npreds = 1;
    [confidence_pitch,labels_pitch] = sort(predictionValues_pitch(:,:,:,1), 3, 'descend') ;    
    confidence_pitch = squeeze(confidence_pitch);
    confidence_pitch = exp(confidence_pitch) / sum(exp(confidence_pitch) );
    labels_pitch = squeeze(labels_pitch);
    pred_pitch = labels_pitch(1:npreds);
    conf_pitch = confidence_pitch(1:npreds);
    
    
    [confidence_heading, labels_heading] = sort(predictionValues_heading(:,:,:,1), 3, 'descend') ;    
    confidence_heading = squeeze(confidence_heading);
    confidence_heading = exp(confidence_heading) / sum(exp(confidence_heading));    
    labels_heading = squeeze(labels_heading);
    pred_heading = labels_heading(1:npreds);     
    conf_heading = confidence_heading(1:npreds);
    
    
    pred = zeros(1,npreds*npreds);
    for p=1:npreds
        for h=1:npreds        
            pred((p-1)*npreds + h) = (pred_heading(h)-1)*15 + pred_pitch(p);
        end
    end
    
    return;    
    
    load('/lab/ilab/30/kai-vp/google_dataset/vp-alexnet-simplenn-obj/mapping.mat');
    mapy = containers.Map(mapping(:,1),mapping(:,3));
    mapx = containers.Map(mapping(:,1),mapping(:,2));
    
    
%     figure; 
    imshow(uint8(iLab_imrescale(imc, [300 300])));
    hold on;
    
 
    nshow = npreds*npreds;
    
    colors = zeros(nshow,3);
    colors(1,:) = [1 0 0 ];
    for c=1:nshow-1
        colors(c+1,:) = [rand rand rand];
    end
    
	b_conf = (conf_heading > 0.9 ) & (conf_pitch > 0.9);
%     if b_conf == true
%         return;
%     end

    
%     colors = {'red', 'green', 'blue'};
    for i=1:nshow
        x = mapx(pred(i));
        y = mapy(pred(i));
%         scatter(x, y ,80, 'MarkerEdgeColor', colors(i,:), 'linewidth', 5);
        ellipse(5 - 100* log(conf_heading),5 - 100* log(conf_pitch),0,x,y, 'r');
        hold on;
    end
    
    

end