function [h_near] = nearfieldChannel(N,lamda,L)
d = lamda/2;
n = 1:N;

alpha_2 = sqrt(1/2) * (randn(1,L)+1i*randn(1,L)); % path gain

for i = 1:L
    r_l(i) = unifrnd(10,80); % Antenna의 center 부터 scatter 까지의 거리
end


for i = 1:L
    theta_l(i) = unifrnd(-1,1); %
end

delta_n = (2*n-N-1)/2; % with n = 1,2,3, ... , N

for l = 1:L
    for n = 1:N
        r_l_n(l,n) = sqrt(r_l(l)^2 + delta_n(n)^2 * d^2 - 2*r_l(l)*delta_n(n)*d*theta_l(l));
    end
end

for l = 1:L
    for n = 1:N
        b_theta_r(l,n) = exp(-1i*(2*pi)*(r_l_n(l,n) - r_l(l)) / lamda);
    end
end

b_theta_r = b_theta_r';

for i = 1:L
    b_theta_r(:,i) = b_theta_r(:,i) * alpha_2(i) ;
end

h_near = sum(b_theta_r,2);

% for i=1:N-1 
%      h_near(i) = sqrt(h_near(i+1)^2 / h_near(i)^2);
% 
% end

end
