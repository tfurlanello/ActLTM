
function [label, labelIdx, score] = cnnClassification(im, model)    
    im_ = single(im) ; % note: 255 range
    im_ = imresize(im_, model.normalization.imageSize(1:2)) ;
    im_ = im_ - model.normalization.averageImage ;

    % run the CNN
    res = vl_simplenn(model, im_) ;

    scores = squeeze(gather(res(end).x)) ;
    
    [sortedScores,sortIndex] = sort(scores(:),'descend');
	label = model.classes.description(sortIndex(1:5));
    label = label';
    labelIdx = sortIndex(1:5);
    score = sortedScores(1:5);

    
%     [bestScore, best] = max(scores);
%     label = model.classes.description{best};
%     labelIdx = best;
%     score = bestScore;
end