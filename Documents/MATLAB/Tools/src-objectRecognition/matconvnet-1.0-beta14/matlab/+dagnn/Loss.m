classdef Loss < dagnn.ElementWise
  properties
    loss = 'softmaxlog'
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      if isempty(params)
          opts.loss = obj.loss;
      else
          opts.isstructured = params{1};
          opts.labelgraph = params{2};
          opts.loss = obj.loss;
      end
        
%       outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss) ;
%      outputs{1} = vl_nnlossStructured(inputs{1}, inputs{2}, inputs{3}, [], 'loss', obj.loss) ;
      outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], opts) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if isempty(params)
          opts.loss = obj.loss;
      else
          opts.isstructured = params{1};
          opts.labelgraph = params{2};
          opts.loss = obj.loss;
      end        
%       derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss) ;
%       derInputs{1} = vl_nnlossStructured(inputs{1}, inputs{2}, inputs{3}, derOutputs{1}, 'loss', obj.loss) ;
      derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, opts) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
