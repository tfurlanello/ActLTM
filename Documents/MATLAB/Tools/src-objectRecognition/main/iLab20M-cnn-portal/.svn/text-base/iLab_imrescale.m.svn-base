function im = iLab_imrescale(im, szRef)
  % resize
  w      = size(im,2) ;
  h      = size(im,1) ;
  factor = [szRef(1)/h, szRef(2)/w];

  factor = max(factor);  
  im = imresize(im, 'scale', factor, ...
                    'method', 'bilinear') ;

  % crop & flip
    w = size(im,2) ;
    h = size(im,1) ;
    
    offset_h = abs(h-szRef(1))/2;
    offset_w = abs(w-szRef(2))/2;
    
    dy = max(1, round(offset_h));
    dx = max(1, round(offset_w));
    
    dyEnd = min(szRef(1)+dy-1, h);
    dxEnd = min(szRef(2)+dx-1, w);   
    
    im = im(dy:dyEnd, dx:dxEnd,:);
    im = imresize(im,  [szRef(1) szRef(2)], 'method', 'bilinear');

end