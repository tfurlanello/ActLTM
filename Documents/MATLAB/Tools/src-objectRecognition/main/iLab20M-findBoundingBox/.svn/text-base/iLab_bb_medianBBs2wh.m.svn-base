function [w,h] = iLab_bb_medianBBs2wh(bkgs, lights, bbs)
% inputs 
%           bkgs - backgrounds
%           lights - lighting conditions
%           bbs - bounding boxes under the specific background and light

% outputs
%           w - width of the object
%           h - height of the object
% method: median filter

    assert(numel(bkgs) == size(bbs,1));
    assert(numel(lights) == size(bbs,1));
    
    unique_lights = unique(lights);
	nLights = numel(unique_lights);

    widths  = zeros(nLights,1);
    heights = zeros(nLights,1);
    
    for l=1:nLights
        bflag      = (lights == unique_lights(l));
        l_w        = bbs(bflag,3);
        l_h        = bbs(bflag,4);
        
        widths(l)  = round(median(l_w));
        heights(l) = round(median(l_h));
    end
        
    w = max(widths);
    h = max(heights);

end