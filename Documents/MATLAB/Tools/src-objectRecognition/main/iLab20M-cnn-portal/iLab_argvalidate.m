function val_args = iLab_argvalidate(opts, args)

if ~isstruct(opts), error('OPTS must be a structure') ; end
if ~iscell(args), args = {args} ; end
 

% convert ARGS into a structure
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
    args.(params{f}) = values{f};
end

% 
fields = fieldnames(opts);
for f=1:numel(params)
    i=find(strcmpi(params{f}, fields)) ;
    if ~isempty(i)
      opts.(fields{i}) = values{f} ;
    end
end

val_args = opts;


end