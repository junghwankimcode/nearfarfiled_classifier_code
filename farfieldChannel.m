function [h_far] = farfieldChannel(N,lamda,L)


alpha_1 = sqrt(1/2) * (randn(1,L)+1i*randn(1,L));

n = 1:N;
d = lamda/2;

for i = 1:L
    theta_l(i) = unifrnd(-1,1);
    %theta_l(i) = unifrnd(-sqrt(3)/2,sqrt(3)/2);
end

% Far-field Channel Model

n = 1:N;

a_l = [];

for l = 1:L
   a_l(:,l) = exp(-1i*(pi)*theta_l(l)*(n-1))'; %streening vector
end

a_l = a_l; 

for i = 1:L
    a_l(:,i) = a_l(:,i) * alpha_1(i);
end

h_far = (sum(a_l,2)); %colum

% for i=1:N-1
% 
%   h_far(i) = sqrt(h_far(i+1)^2 / h_far(i)^2);
% 
% end

end
