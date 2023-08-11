function [U,S,V] = makeHankel(observation)

    l = length(observation);
    
    r = ceil(l/2);
    c = l - r + 1;
    m = zeros(r,c);
    for kk=1:r
        m(kk,:) = (observation(kk:kk+c-1));
    end

    [U,S,V] = svd(m);
   
    
end
