function obj = setobj(net)
% LOADOBJ  Initialize a DagNN object from a structure.
%   OBJ = LOADOBJ(S) initializes a DagNN objet from the structure
%   S. It is the opposite of S = OBJ.SAVEOBJ().

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isa(net, 'dagnn.DagNN')
  obj = dagnn.DagNN() ;
  if isprop(net, 'layers')
      for l = 1:numel(net.layers)
        block = net.layers(l).block;
        obj.addLayer(...
          net.layers(l).name, ...
          block, ...
          net.layers(l).inputs, ...
          net.layers(l).outputs, ...
          net.layers(l).params) ;
      end
  end
  if isprop(net, 'params')
    for f = setdiff(fieldnames(net.params)','name')
      f = char(f) ;
      for i = 1:numel(net.params)
        p = obj.getParamIndex(net.params(i).name) ;
        obj.params(p).(f) = net.params(i).(f) ;
      end
    end
  end
  if isprop(net, 'vars')
    for f = setdiff(fieldnames(net.vars)','name')
      f = char(f) ;
      for i = 1:numel(net.vars)
        p = obj.getVarIndex(net.vars(i).name) ;
        obj.vars(p).(f) = net.vars(i).(f) ;
      end
    end
  end
  for f = setdiff(fieldnames(net)', {'vars','params','layers'})
    f = char(f) ;
    obj.(f) = net.(f) ;
  end
else
    error('the input should be a dagnn.DagNN instance\n');

end

end
