function opts = iLab_arg2struct(args)
    % convert ARGS into a structure
    if nargin == 0 || ~exist('args', 'var') || isempty(args)
        opts = struct([]); return;
    end
    if ~iscell(args), args = {args} ; end
    ai = 1 ;
    params = {} ;
    values = {} ;
    while ai <= length(args)
      if isstr(args{ai})
        params{end+1} = args{ai} ; ai = ai + 1 ;
        values{end+1} = args{ai} ; ai = ai + 1 ;
      elseif isstruct(args{ai}) ;
        params = horzcat(params, fieldnames(args{ai})') ;
        values = horzcat(values, struct2cell(args{ai})') ;
        ai = ai + 1 ;
      else
        error('Expected either a param-value pair or a structure') ;
      end
    end

    
    for f=1:numel(params)
        opts.(params{f}) = values{f};
    end

end