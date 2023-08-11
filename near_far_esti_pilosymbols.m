clear all
clc;

TrainSet = 1e4;

% P=16;%pilot symbols
PVec=[4:4:64];


L = 6; % The number of all path components

N = 512; % number of Antenna
% stepsize=floor(N/(P-1));

lamda = 0.01; % wavelength

% trainSNR = 10;
trainSNRVec=[-10:5:25];

dd = 2; 
str = 64;

trainSNR = 10;
SNRmargin=0;

TestSet = 1e4;

iterMax = 30;

for indexN = 1:length(PVec)

    P= PVec(indexN);
    stepsize=floor(N/(P-1));

trainAns = [];

testAns  = [];

trainData1 = []; % Regacy Algorithm

testData1  = [];


trainData2 = []; % Hankelization Algorithm

testData2  = [];

trainData3 = []; % FFT Algorithm

testData3 = [];

parfor k=1:TrainSet

    whoRU = randi(2,1);

    if whoRU == 1 % Near-field
      
      ansVec = [1 0];

      xx = nearfieldChannel(N,lamda,L);
      xx = xx.';
      xx=xx(1:stepsize:N);
      xx= xx / norm(xx);
      xn = awgn(xx,trainSNR,'measured');

      trainData1 = [trainData1 transpose(abs((xn)))];

      [Un, Sn, Vn] = makeHankel(xn);

      trainData2 = [trainData2 (diag(Sn))]; %singular value들 normalize한 행렬

      fft_xn = abs(fft(xn));

      trainData3 = [trainData3 transpose(fft_xn)];

      trainAns = [trainAns transpose(ansVec)];

    else  % Far-Field

    ansVec = [0 1];  

      xx = farfieldChannel(N,lamda,L);
      xx = xx.';
      xx=xx(1:stepsize:N);
      xx= xx / norm(xx);
      xn = awgn(xx,trainSNR,'measured');

      trainData1 = [trainData1 transpose(abs((xn)))];

      [Un, Sn, Vn] = makeHankel(xn);

      trainData2 = [trainData2 (diag(Sn))];

      fft_xn = abs(fft(xn));

      trainData3 = [trainData3 transpose(fft_xn)];

      trainAns = [trainAns transpose(ansVec)];

    end

end


net1 = fitnet(str*ones(1,dd));

net1.trainFcn = 'trainscg';

net1 = train(net1,(trainData1), (trainAns),'useGPU','yes');


net2 = fitnet(str*ones(1,dd));

net2.trainFcn = 'trainscg';

net2 = train(net2,(trainData2), (trainAns),'useGPU','yes');

net3 = fitnet(str*ones(1,dd));

net3.trainFcn = 'trainscg';

net3 = train(net3,trainData3,trainAns,'useGPU','yes');


parfor k=1:TestSet

  whoRU = randi(2,1);

     testSNR  = trainSNR + rand(1)*SNRmargin - SNRmargin/2;

    if whoRU == 1 % Near-field

      ansVec = [1 0];    

      xx = nearfieldChannel(N,lamda,L);
      xx = xx.';
      xx=xx(1:stepsize:N);
      xx= xx / norm(xx);
      xn = awgn(xx,trainSNR,'measured');

      testData1 = [testData1 transpose(abs((xn)))];

      [Un, Sn, Vn] = makeHankel(xn);

      testData2 = [testData2 (diag(Sn))];

      fft_xn = abs(fft(xn));

      testData3 = [testData3 transpose(fft_xn)];

      testAns = [testAns transpose(ansVec)];

    else  % Far-Field

      ansVec = [0 1];
     
      xx = farfieldChannel(N,lamda,L);
      xx = xx.';
      xx=xx(1:stepsize:N);
      xx= xx / norm(xx);
      xn = awgn(xx,trainSNR,'measured');

      testData1 = [testData1 transpose(abs((xn)))];

      [Un, Sn, Vn] = makeHankel(xn);

      testData2 = [testData2 (diag(Sn))];

      fft_xn = abs(fft(xn));

      testData3 = [testData3 transpose(fft_xn)];

      testAns = [testAns transpose(ansVec)];
      
    end

end

testResult1 = net1(testData1);

testResult1_mat = zeros(2,TestSet);

[ss ii] = max(testResult1);

for k=1:TestSet

testResult1_mat(ii(k),k)=1;

end

testResult2 = net2(testData2);

testResult2_mat = zeros(2,TestSet);

[ss ii] = max(testResult2);

for k=1:TestSet

testResult2_mat(ii(k),k)=1;

end

testResult3 = net3(testData3);
testResult3_mat = zeros(2,TestSet);
[ss ii] = max(testResult3);
for k=1:TestSet
testResult3_mat(ii(k),k)=1;
end


detectionRate1(indexN) = sum(sum(testResult1_mat.*testAns)) / TestSet;

detectionRate2(indexN) = sum(sum(testResult2_mat.*testAns)) / TestSet;

detectionRate3(indexN) = sum(sum(testResult3_mat.*testAns)) / TestSet;

end

figure('Color','w')

plot(PVec, detectionRate1); hold on

plot(PVec, detectionRate2); hold on

plot(PVec, detectionRate3); hold on

xlabel('Number of pilot symbols')

ylabel('Detection rate')

legend('Deep learning-based (legacy)','Hankelization Algorithm','Deep learning-based (FFT abs)','location','best')

xticks(PVec)

xlim([4 64])

ylim([0.0 1])

grid on

hold off
