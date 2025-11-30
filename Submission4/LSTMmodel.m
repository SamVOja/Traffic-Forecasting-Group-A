% LSTM Model

numTimeSteps = 24; % 1 day lookback window

XTrain_lstm = {};
YTrain_lstm = [];
for i = 1:length(train_target) - numTimeSteps
    XTrain_lstm{end+1} = train_features(i:i+numTimeSteps-1, :)';
    YTrain_lstm(end+1) = train_target(i+numTimeSteps);
end
XTrain_lstm = XTrain_lstm';
YTrain_lstm = YTrain_lstm';

XTest_lstm = {};
YTest_lstm = [];
for i = 1:length(test_target) - numTimeSteps
    XTest_lstm{end+1} = test_features(i:i+numTimeSteps-1, :)';
    YTest_lstm(end+1) = test_target(i+numTimeSteps);
end
XTest_lstm = XTest_lstm';
YTest_lstm = YTest_lstm';

layers = [ ...
    sequenceInputLayer(numFeatures)  
    lstmLayer(30, 'OutputMode', 'last')  
    dropoutLayer(0.2)
    fullyConnectedLayer(15)             
    reluLayer()
    fullyConnectedLayer(1)
    regressionLayer()];

options = trainingOptions('adam', ...
    'MaxEpochs', 40, ...                         
    'InitialLearnRate', 0.001, ...                  
    'GradientThreshold', 1, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 20, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationData', {XTest_lstm, YTest_lstm}, ... 
    'ValidationFrequency', 100, ...
    'Verbose', 1, ...
    'Plots', 'training-progress');

% Train
fprintf('Training LSTM\n');
net = trainNetwork(XTrain_lstm, YTrain_lstm, layers, options);
YPredTest = predict(net, XTest_lstm);

% Calculate Performance Metrics
testMSE = mean((YTest_lstm - YPredTest).^2);
testRMSE = sqrt(testMSE);
testMAE = mean(abs(YTest_lstm - YPredTest));

fprintf('\n=== LSTM Test Performance ===\n');
fprintf('MSE:  %.4f\n', testMSE);
fprintf('RMSE: %.4f\n', testRMSE);
fprintf('MAE:  %.4f\n', testMAE);

test_mean = mean(YTest_lstm);
ss_total = sum((YTest_lstm - test_mean).^2);
ss_residual = sum((YTest_lstm - YPredTest).^2);
r_squared = 1 - (ss_residual / ss_total);

fprintf('R-squared: %.4f\n', r_squared);

% Plot results
figure;
plot(YTest_lstm, 'b-', 'LineWidth', 2, 'DisplayName', 'Actual');
hold on;
plot(YPredTest, 'r--', 'LineWidth', 1.5, 'DisplayName', 'LSTM Predicted');
title("LSTM - Test Partition: Actual vs Predicted");
legend;
xlabel('Time Step');
ylabel('Traffic Volume');
grid on;