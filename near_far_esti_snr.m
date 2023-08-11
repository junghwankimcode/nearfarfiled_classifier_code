clear all
clc;

TrainSet = 1e3;

P=32;%pilot symbols

L = 1; % The number of all path components

N = 512; % number of Antenna
stepsize=floor(N/(P-1));

lamda = 0.01; % wavelength

% trainSNR = 10;
trainSNRVec=[-10:5:25];

dd = 2; 
str = 64;

SNRmargin=0;

TestSet = 1e3;

iterMax = 30;

for indexN = 1:length(trainSNRVec)

    trainSNR = trainSNRVec(indexN);

trainAns = [];

testAns  = [];

trainData1 = []; % legacy angle

testData1  = [];


trainData2 = []; % Hankelization Algorithm

testData2  = [];

trainData3 = []; % fft abs

testData3 = [];

trainData4=[]; %legacy angle
testData4=[];

trainData5=[]; %fft angle
testData5=[];

trainData6=[]; %legacy angle & abs mean
testData6=[];

trainData7=[]; %fft angle & abs mean
testData7=[];



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

      fft_xn = (fft(xn));

      trainData3 = [trainData3 transpose(abs(fft_xn))];

      trainData4 = [trainData4 transpose(angle(xn))];

      trainData5 = [trainData5 transpose(angle(fft_xn))];

      trainData6 = [trainData6 transpose(max(angle(xn),abs(xn)))];

      trainData7 = [trainData7 transpose(max(angle(fft_xn),abs(fft_xn)))];

      
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

      fft_xn = (fft(xn));

      trainData3 = [trainData3 transpose(abs(fft_xn))];

      trainData4 = [trainData4 transpose(angle(xn))];

      trainData5 = [trainData5 transpose(angle(fft_xn))];

      trainData6 = [trainData6 transpose(max(angle(xn),abs(xn)))];

      trainData7 = [trainData7 transpose(max(angle(fft_xn),abs(fft_xn)))];

      

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


net4 = fitnet(str*ones(1,dd));

net4.trainFcn = 'trainscg';

net4 = train(net4,trainData4,trainAns,'useGPU','yes');


net5 = fitnet(str*ones(1,dd));

net5.trainFcn = 'trainscg';

net5 = train(net5,trainData5,trainAns,'useGPU','yes');


net6 = fitnet(str*ones(1,dd));

net6.trainFcn = 'trainscg';

net6 = train(net6,trainData6,trainAns,'useGPU','yes');


net7 = fitnet(str*ones(1,dd));

net7.trainFcn = 'trainscg';

net7 = train(net7,trainData7,trainAns,'useGPU','yes');


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

      fft_xn = (fft(xn));

      testData3 = [testData3 transpose(abs(fft_xn))];

      testData4 = [testData4 transpose(angle(xn))];

      testData5 = [testData5 transpose(angle(fft_xn))];

      testData6 = [testData6 transpose(max(angle(xn),abs(xn)))];

      testData7 = [testData7 transpose(max(angle(fft_xn),abs(fft_xn)))];


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

      fft_xn = (fft(xn));

      testData3 = [testData3 transpose(abs(fft_xn))];

      testData4 = [testData4 transpose(angle(xn))];

      testData5 = [testData5 transpose(angle(fft_xn))];

      testData6 = [testData6 transpose(max(angle(xn),abs(xn)))];

      testData7 = [testData7 transpose(max(angle(fft_xn),abs(fft_xn)))];


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

testResult4 = net4(testData4);
testResult4_mat = zeros(2,TestSet);
[ss ii] = max(testResult4);
for k=1:TestSet
testResult4_mat(ii(k),k)=1;
end

testResult5 = net5(testData5);
testResult5_mat = zeros(2,TestSet);
[ss ii] = max(testResult5);
for k=1:TestSet
testResult5_mat(ii(k),k)=1;
end

testResult6 = net6(testData6);
testResult6_mat = zeros(2,TestSet);
[ss ii] = max(testResult6);
for k=1:TestSet
testResult6_mat(ii(k),k)=1;
end

testResult7 = net7(testData7);
testResult7_mat = zeros(2,TestSet);
[ss ii] = max(testResult7);
for k=1:TestSet
testResult7_mat(ii(k),k)=1;
end


detectionRate1(indexN) = sum(sum(testResult1_mat.*testAns)) / TestSet;

detectionRate2(indexN) = sum(sum(testResult2_mat.*testAns)) / TestSet;

detectionRate3(indexN) = sum(sum(testResult3_mat.*testAns)) / TestSet;

detectionRate4(indexN) = sum(sum(testResult4_mat.*testAns)) / TestSet;

detectionRate5(indexN) = sum(sum(testResult5_mat.*testAns)) / TestSet;

detectionRate6(indexN) = sum(sum(testResult6_mat.*testAns)) / TestSet;

detectionRate7(indexN) = sum(sum(testResult7_mat.*testAns)) / TestSet;


end

figure('Color','w')

plot(trainSNRVec, detectionRate1, 'k'); hold on

plot(trainSNRVec, detectionRate4, 'g'); hold on

plot(trainSNRVec, detectionRate6,'b'); hold on

plot(trainSNRVec, detectionRate3,'c'); hold on

plot(trainSNRVec, detectionRate5,'y'); hold on

plot(trainSNRVec, detectionRate7,'m'); hold on

plot(trainSNRVec, detectionRate2,'r'); hold on


xlabel('SNR[dB]')

ylabel('Detection rate')

legend('Deep learning-based (legacy abs)','Deep learning-based (legacy angle)','Deep learning-based (legacy max of angle & abs)', ...
    'Deep learning-based (FFT abs)','Deep learning-based (FFT angle)','Deep learning-based (fft max of angle & abs)', ...
    'Hankelization Algorithm','location','best')


xticks(trainSNRVec)

xlim([-10 25])

ylim([0.0 1])

grid on

hold off
