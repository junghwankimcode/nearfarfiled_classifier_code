clear all
clc;

TrainSet = 1e4;

P=32;%pilot symbols

L = 6; % The number of all path components

N = 512; % number of Antenna
stepsize=floor(N/(P-1));

lamda = 0.01; % wavelength

trainSNRVec=[10];

dd = 2; 
str = 16;

SNRmargin=0;

TestSet = 5e4;

iterMax = 30;

for indexN = 1:length(trainSNRVec)

    trainSNR = trainSNRVec(indexN);

trainAns = [];

testAns  = [];

trainData1 = []; % legacy abs

testData1  = [];


trainData2 = []; % Hankelization Algorithm

testData2  = [];

trainData3 = []; % fft abs

testData3 = [];



parfor k=1:TrainSet

    whoRU = randi(2,1);

    if whoRU == 1 % Near-field
      
      ansVec = [1 0];

      xx = farfieldChannel(N,lamda,L);
      xx = xx.';
      xx=xx(1:stepsize:N);
      xx= xx / norm(xx);
      xn = awgn(xx,trainSNR,'measured');

      trainData1 = [trainData1 transpose(abs((xn)))];

      [Un, Sn, Vn] = makeHankel(xn);

      trainData2 = [trainData2 (diag(Sn))]; %singular value들 normalize한 행렬

      fft_xn = (fft(xn));

      trainData3 = [trainData3 transpose(abs(fft_xn))];


      
      trainAns = [trainAns transpose(ansVec)];

    else  % Far-Field

    ansVec = [0 1];  

      xx = nearfieldChannel(N,lamda,L);
      xx = xx.';
      xx=xx(1:stepsize:N);
      xx= xx / norm(xx);
      xn = awgn(xx,trainSNR,'measured');

      trainData1 = [trainData1 transpose(abs((xn)))];

      [Un, Sn, Vn] = makeHankel(xn);

      trainData2 = [trainData2 (diag(Sn))];

      fft_xn = (fft(xn));

      trainData3 = [trainData3 transpose(abs(fft_xn))];

      trainAns = [trainAns transpose(ansVec)];

    end

end


options = trainingOptions('sgdm', ...
     'InitialLearnRate', 1e-5, ...
    'MaxEpochs', 1e3, ...
    'MiniBatchSize', 8, ...
   'Shuffle', 'every-epoch', ...
   'Momentum',0,...
    'Verbose', 0,...
    Plots="training-progress");


layers = [sequenceInputLayer(P)
        fullyConnectedLayer(str)
        tanhLayer
        fullyConnectedLayer(str)
        tanhLayer
        fullyConnectedLayer(2)
        regressionLayer];

layers2 = [sequenceInputLayer(P/2)
        fullyConnectedLayer(str)
        tanhLayer
        fullyConnectedLayer(str)
        tanhLayer
        fullyConnectedLayer(2)
        regressionLayer];



net1 = trainNetwork(trainData1, trainAns, layers, options);


net2 = trainNetwork(trainData2, trainAns, layers2, options);


net3 = trainNetwork(trainData3, trainAns, layers, options);



parfor k=1:TestSet

  whoRU = randi(2,1);

     testSNR  = trainSNR + rand(1)*SNRmargin - SNRmargin/2;

    if whoRU == 1 % Near-field

      ansVec = [1 0];    

      xx = farfieldChannel(N,lamda,L);
      xx = xx.';
      xx=xx(1:stepsize:N);
      xx= xx / norm(xx);
      xn = awgn(xx,testSNR,'measured');

      testData1 = [testData1 transpose(abs((xn)))];

      [Un, Sn, Vn] = makeHankel(xn);

      testData2 = [testData2 (diag(Sn))];

      fft_xn = (fft(xn));

      testData3 = [testData3 transpose(abs(fft_xn))];

      
      testAns = [testAns transpose(ansVec)];

    else  % Far-Field

      ansVec = [0 1];
     
      xx = nearfieldChannel(N,lamda,L);
      xx = xx.';
      xx=xx(1:stepsize:N);
      xx= xx / norm(xx);
      xn = awgn(xx,testSNR,'measured');

      testData1 = [testData1 transpose(abs((xn)))];

      [Un, Sn, Vn] = makeHankel(xn);

      testData2 = [testData2 (diag(Sn))];

      fft_xn = (fft(xn));

      testData3 = [testData3 transpose(abs(fft_xn))];



      testAns = [testAns transpose(ansVec)];
      
    end

end

testResult1 = predict(net1,testData1);
testResult1=double(testResult1);

testResult1_mat = zeros(2,TestSet);

[ss ii] = max(testResult1);

for k=1:TestSet

testResult1_mat(ii(k),k)=1;

end

testResult2 = predict(net2,testData2);
testResult2=double(testResult2);

testResult2_mat = zeros(2,TestSet);

[ss ii] = max(testResult2);

for k=1:TestSet

testResult2_mat(ii(k),k)=1;

end

testResult3 = predict(net3,testData3);
testResult3=double(testResult3);

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
[c,cm,ind,per]=confusion(testAns, testResult2_mat);
 m=cm;
 classLabels={'Near field';'Far field'};
 cm=confusionchart(m,classLabels, 'ColumnSummary', 'column-normalized','RowSummary','row-normalized')
 sortClasses(cm, ["Near field", "Far field"])
 title('Proposed neural network')
 xlabel('Predicted Class')
 ylabel('True Class')
