function [crop, winsize, cropbox] = iLab_bb_intelligentClipper(im, winsize, objCenter)
% inputs
%           im - input image
%           winsize - the size of the clipping window
%           objCenter - the approximate center of the object
% ouputs
%           crop - the cropped patch
        [imh, imw, ~] = size(im);
        polyim = rectangle2polygon([1 1 imw imh]);
        
        b = (winsize <= imh) & (winsize <= imw); 
        if ~b
            winsize = min([imh imw]);
        end
        
        xcenter = objCenter(1);
        ycenter = objCenter(2);
        
        xs = xcenter - round(winsize/2);
        ys = ycenter - round(winsize/2);
        
        xe = xs + winsize - 1;
        ye = ys + winsize - 1;
        
        rec_crop = [xs ys winsize winsize];
        polycrop = rectangle2polygon(rec_crop);
        
        % first clip to get the union
        type = 1;
        intersection     = PolygonClip(polycrop, polyim, type);          
        rec_intersection = polygon2rectangle(intersection);
        
        %% in most cases
        if sum(rec_crop == rec_intersection) == 4
            crop = im(ys:ye, xs:xe, :);
            cropbox = [xs ys winsize winsize];
            return;
        end
        
        %% vertically shifting
        winw = rec_intersection(3);
        winh = rec_intersection(4);        
        xs = rec_intersection(1);
        ys = rec_intersection(2);
        xe = xs + winw - 1;
        ye = ys + winh - 1;
        
        if winh < winsize
            if ys == 1 % move down
                rec_intersection = [xs ys winw winsize];
            elseif ye == imh % move up
                rec_intersection = [xs ye-winsize+1 winw winsize];
            end
        end
        
        %% horizontally shifting
        winw = rec_intersection(3);
        winh = rec_intersection(4);        
        xs = rec_intersection(1);
        ys = rec_intersection(2);
        xe = xs + winw - 1;
        ye = ys + winh - 1;
        
        if winw < winsize
            if xs == 1 % move right
                rec_intersection = [xs ys winsize winh];
            elseif xe == imw % move left
                rec_intersection = [xe-winsize+1 ys winsize winh];
            end
        end
                
        %% cropping
        winw    =   rec_intersection(3);
        winh    =   rec_intersection(4);        
        xs      =   rec_intersection(1);
        ys      =   rec_intersection(2);
        xe      =   xs + winw - 1;
        ye      =   ys + winh - 1;
        
        assert( (winw == winsize) && (winh == winsize));        
        crop = im(ys:ye, xs:xe,:);
        cropbox = [xs ys winw winh];
        

end