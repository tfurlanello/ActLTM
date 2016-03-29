function labelgraph = vp_gen_labelgraph(npitch, nheading)
    
    nlabels = npitch * nheading;
    labelgraph = zeros(nlabels, nlabels);
    
    
    for p=1:npitch
        for h=1:nheading
            
            idx_ref = (h-1)*npitch + p;
            
            idx_nei = [];
            cnt = 0;
            for m= max(1, h-1):min(h+1, nheading)
                for n=max(1,p-1):min(p+1, npitch)
                    cnt = cnt + 1;
                    idx_nei = cat(1, idx_nei, (m-1)*npitch + n);
                end
            end
            
            wei_ref = 0.6;
            wei_nei = (1-wei_ref)/(numel(idx_nei)-1);
            
            
            for n=1:numel(idx_nei)
                if idx_nei(n) ~= idx_ref
                    labelgraph(idx_ref, idx_nei(n)) = wei_nei;
                else
                    labelgraph(idx_ref, idx_nei(n)) = wei_ref;
                end
            end
            
            
        end
    end


end