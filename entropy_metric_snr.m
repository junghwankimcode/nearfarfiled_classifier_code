clear all;

clc;

% 이론적 해석

nDrop = 1e4;

trainNoAvec = [512]; % 안테나 수

trainSNRvec = [-10:5:25];

Amp = 1e4;

L = 6;
P=32;
lamda = 0.01;
stepsize=floor(trainNoAvec/(P-1));

for indexNoA = 1:length(trainNoAvec)
    trainNoA = trainNoAvec(indexNoA);

for indexSNR = 1:length(trainSNRvec)

    trainSNR = trainSNRvec(indexSNR);
   
    
    trainData_x1 = [];
    trainData_x2 = [];
    trainData_x3 = [];
    trainData_y1 = [];
    trainData_y2 = [];
    trainData_y3 = [];


parfor indexDrop = 1:nDrop


xx = nearfieldChannel(trainNoA,lamda,L);
xx= xx.';
xx=xx(1:stepsize:trainNoAvec);
xx=xx/norm(xx);
xn = awgn(xx,trainSNR,'measured');

trainData_x1 = [trainData_x1 transpose(((xn)))];

[Un, Sx, Vn] = makeHankel(xn);

trainData_x2 = [trainData_x2 (diag(Sx))]; %singular value들 normalize한 행렬

result = (fft(xn));

trainData_x3 = [trainData_x3 transpose(result)];


yy = farfieldChannel(trainNoA,lamda,L);
yy = yy.';
yy=yy(1:stepsize:trainNoAvec);
yy = yy/norm(yy);
yn = awgn(yy,trainSNR,'measured');

trainData_y1 = [trainData_y1 transpose((yn))];

[Uy,Sy,Vy] = makeHankel(yn);

trainData_y2 = [trainData_y2 (diag(Sy))];

result = (fft(yn));

trainData_y3 = [trainData_y3 transpose(result)];


end


    %x1y1MI(indexSNR) = 0.5*(mutInfoPortion(round(Amp*real(trainData_x1)), round(Amp*real(trainData_y1))) + mutInfoPortion(round(Amp*imag(trainData_x1)), round(Amp*imag(trainData_y1))));
    % real+imag 이기에 0.5 곱함 
    x1y1MI(indexSNR) = mutInfoPortion(round(Amp*abs(trainData_x1)), round(Amp*abs(trainData_y1)));

    SxSyMI(indexSNR) = mutInfoPortion(round(Amp*trainData_x2), round(Amp*trainData_y2));

    %fxfyMI(indexSNR) = 0.5* (mutInfoPortion(round(Amp*real(trainData_x3)),round(Amp*real(trainData_y3))) + mutInfoPortion(round(Amp*imag(trainData_x3)),round(Amp*imag(trainData_y3))));
    fxfyMI(indexSNR) = mutInfoPortion(round(Amp*abs(trainData_x3)),round(Amp*abs(trainData_y3))); 

    %x1y1Me(indexSNR) = 0.5*(metric(round(Amp*real(trainData_x1)), round(Amp*real(trainData_y1))) + metric(round(Amp*imag(trainData_x1)), round(Amp*imag(trainData_y1))));
    x1y1Me(indexSNR) = metric(round(Amp*abs(trainData_x1)), round(Amp*abs(trainData_y1)));

    SxSyMe(indexSNR) = metric(round(Amp*trainData_x2), round(Amp*trainData_y2));

    %fxfyMe(indexSNR) = 0.5* (metric(round(Amp*real(trainData_x3)),round(Amp*real(trainData_y3))) + metric(round(Amp*imag(trainData_x3)),round(Amp*imag(trainData_y3))));
    fxfyMe(indexSNR) = metric(round(Amp*abs(trainData_x3)),round(Amp*abs(trainData_y3)));


end


figure('Color','w')

plot(trainSNRvec, (x1y1MI), 'go-'); hold on

plot(trainSNRvec, (SxSyMI), 'r*:'); hold on

plot(trainSNRvec, (fxfyMI), 'bx-'); hold on

grid on

xlim([-10 25])
xticks(trainSNRvec)

xlabel('SNR[dB]')

ylabel('Mutual information / Entropy of near-field data')

legend("Deep learning based (legacy)", "Hankelized Algorithm" , 'Deep learning based (fft abs)');



figure('Color','w')

plot(trainSNRvec,x1y1Me,'go-'); hold on

plot(trainSNRvec, (SxSyMe), 'r*:'); hold on

plot(trainSNRvec, (fxfyMe), 'bx-'); hold on

grid on

xlim([-10 25])
xticks(trainSNRvec)

xlabel('SNR[dB]')

ylabel('Symmetric uncertainty')

legend("Deep learning based (legacy)", "Hankelized Algorithm" , 'Deep learning based (fft abs)');


end

hold off


%%

function [P] = mutInfoPortion(x, y) % 지금 여기서 input은 matrix


assert(numel(x) == numel(y)); % 요소 개수가 같아야함 다르면 오류 발생

n = numel(x); % n = x,y의 요소 개수

x = reshape(x,1,n);  % matrix를 vector로 

y = reshape(y,1,n);  % "

l = min(min(x),min(y)); % x의 최소와 y의 최소를 비교해 가장 작은 값을 비교

x = x-l+1;

y = y-l+1;

k = max(max(x),max(y));

idx = 1:n;

Mx = sparse(idx,x,1,n,k,n);

My = sparse(idx,y,1,n,k,n);

Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y

Hxy = -dot(Pxy,log2(Pxy));

Px = nonzeros(mean(Mx,1)); % low vector의 평균값

Py = nonzeros(mean(My,1));

% entropy of Py and Px

Hx = -dot(Px,log2(Px));

Hy = -dot(Py,log2(Py));

% mutual information

z = Hx+Hy-Hxy;

z = max(0,z);

P = z / Hx;
end

%%
function [P] = metric(x, y) % 지금 여기서 input은 matrix


assert(numel(x) == numel(y)); % 요소 개수가 같아야함 다르면 오류 발생

n = numel(x); % n = x,y의 요소 개수

x = reshape(x,1,n);  % matrix를 vector로 

y = reshape(y,1,n);  % "

l = min(min(x),min(y)); % x의 최소와 y의 최소를 비교해 가장 작은 값을 비교

x = x-l+1;

y = y-l+1;

k = max(max(x),max(y));

idx = 1:n;

Mx = sparse(idx,x,1,n,k,n);

My = sparse(idx,y,1,n,k,n);

Pxy = nonzeros(Mx'*My/n); %joint distribution of x and y

Hxy = -dot(Pxy,log2(Pxy));

Px = nonzeros(mean(Mx,1)); % low vector의 평균값

Py = nonzeros(mean(My,1));

% entropy of Py and Px

Hx = -dot(Px,log2(Px));

Hy = -dot(Py,log2(Py));

% mutual information

z = Hx+Hy-Hxy;

z = max(0,z);

P = 2*z/(Hx+Hy);

end
