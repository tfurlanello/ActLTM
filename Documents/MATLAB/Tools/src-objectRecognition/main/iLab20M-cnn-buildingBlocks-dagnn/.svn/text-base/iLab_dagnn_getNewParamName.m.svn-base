function name = iLab_dagnn_getNewParamName(obj, prefix)
% --------------------------------------------------------------------
    t = 0 ;
    name = prefix ;
    while any(strcmp(name, {obj.params.name}))
      t = t + 1 ;
      name = sprintf('%s%d', prefix, t) ;
    end
    
    while any(strcmp(name, {obj.layers.name}))
      t = t + 1 ;
      name = sprintf('%s%d', prefix, t) ;
    end
    
    while any(strcmp(name, {obj.vars.name}))
      t = t + 1 ;
      name = sprintf('%s%d', prefix, t) ;
    end

end