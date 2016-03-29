% test multi-dimensional array

tic
X = rand(3,4,100,200);
opts.loss = 'crossentropy';

 



%% (1) weight matrix
nclasses = size(X,3);
labelAdjacency = logical(randi(2, [nclasses, nclasses])-1);
labelInteraction = rand(nclasses);

inputSize = [size(X,1) size(X,2) size(X,3) size(X,4)] ;
c = randi(inputSize(3), 1, inputSize(4));

% Form 1: C has one label per image. In this case, get C in form 2 or
% form 3.
c = gather(c) ;
if numel(c) == inputSize(4)
  c = reshape(c, [1 1 1 inputSize(4)]) ;
  c = repmat(c, inputSize(1:2)) ;
end

nCategories = size(labelInteraction,1);
weights = zeros(size(X));
labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)] ;

assert(isequal(inputSize(1:2), labelSize(1:2)));
assert(inputSize(4) == labelSize(4));
assert(labelSize(3) == 1);
assert(inputSize(3) == nCategories);
for i=1:labelSize(1)
    for j=1:labelSize(2)
        for k=1:labelSize(4)
            weights(i,j,:,k) = labelInteraction(c(i,j,1,k),:);
        end
    end
end


%% sanity check of classification errors
tic
predictions = zeros(size(X));
for i=1:nCategories
  i_weight = reshape(labelInteraction(i,:), [1 1 nCategories 1]);
  i_weight = repmat(i_weight, [inputSize(1) inputSize(2) 1 inputSize(4)]);
  predictions(:,:,i,:) = sum(X .* i_weight,3);
end
[~,chat] = max(predictions,[],3) ;
t = single(c ~= chat) ; 
toc;

% tic
% i_weights = reshape(labelInteraction', [1 1 nCategories nCategories 1]);
% i_weights = repmat(i_weights, [inputSize(1) inputSize(2) 1 1 inputSize(4)]);
% i_X = reshape(X, [inputSize(1) inputSize(2) inputSize(3) 1 inputSize(4)]);
% i_X = repmat(i_X, [1 1 1 nCategories 1]);
% tmp = sum(i_X .* i_weights,3);
% [~, chat2] = max(tmp, [], 4);
% i_c = reshape(c, [inputSize(1) inputSize(2)  1 1 inputSize(4)]);
% t2 = single(i_c ~= chat2);
% toc;

%% (2) cross-entropy loss
dzdy = [];
if  isempty(dzdy)
  switch lower(opts.loss)
    case 'crossentropy'
      Xmax = max(X,[],3) ;
      ex = exp(bsxfun(@minus, X, Xmax)) ;
      t = repmat(Xmax, [1 1 inputSize(3) 1]) + repmat(log(sum(ex,3)), [1 1 inputSize(3) 1]) - X ;
      t = t .* weights;
      otherwise
  end
end
toc

%% (3) get the derivative
tic
dzdy = 1;
 if  ~isempty(dzdy)
%   dzdy = dzdy * instanceWeights ;
  switch lower(opts.loss)
    case 'crossentropy'
      Xmax = max(X,[],3) ;
      ex = exp(bsxfun(@minus, X, Xmax)) ;
      Y = bsxfun(@rdivide, ex, sum(ex,3)) ;
      Y = repmat(sum(weights,3), [1 1 inputSize(3) 1]) .* Y - weights;      
      Y = bsxfun(@times, dzdy, Y) ;
      otherwise
  end
 end
toc

% tic
% sanity test 
% summation = 0;
% for i=1:inputSize(1)
%     for j=1:inputSize(2)
%         for k=1:inputSize(4)
%             x = squeeze(X(i,j,:,k));
%             tmp = log(sum(exp(x))) - x;
%             w = weights(i,j,:,k);
%             res = w(:) .* tmp(:);
%             
%             ref = squeeze(t(i,j,:,k));
%             summation = summation + sum(res - ref);
%             
%         end
%     end
% end
% toc




%sanity test of derivative
% for i=1:inputSize(1)
%     for j=1:inputSize(2)
%         for k=1:inputSize(4)
%             x = squeeze(X(i,j,:,k));
%             dx = exp(x) / sum(exp(x));
%             w = squeeze(weights(i,j,:,k));
%             dx = dx * sum(w);
%             dx = dx(:) - w(:);
%             
%             d = dx - squeeze(Y(i,j,:,k));
%             
%             summation = summation + sum(d);
%             
%             
%         end
%     end
% end
% 














