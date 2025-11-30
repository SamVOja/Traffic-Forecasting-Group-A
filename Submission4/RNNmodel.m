% RNN MODEL

% Single-step prediction (current features -> next traffic volume)
X_train_rnn = {};
T_train_rnn = {};
for i = 1:length(train_target)-1
    X_train_rnn{i} = train_features(i, :)';
    T_train_rnn{i} = train_target(i+1);
end

X_test_rnn = {};
T_test_rnn = {};
for i = 1:length(test_target)-1
    X_test_rnn{i} = test_features(i, :)';
    T_test_rnn{i} = test_target(i+1);
end

net = layrecnet(1:2, 15);
net.inputs{1}.size = numFeatures;
net.trainParam.epochs = 30;
net.trainParam.showWindow = true;
net.divideFcn = 'dividetrain';  
net.divideMode = 'sample';    
net.divideParam.trainInd = 1:length(X_train_rnn);
net.divideParam.valInd = [];
net.divideParam.testInd = [];

trainedNet = train(net, X_train_rnn, T_train_rnn);

% Make predictions
YCalPred = trainedNet(X_train_rnn);
YCalPred = cell2mat(YCalPred);
Y = trainedNet(X_test_rnn);
Y = cell2mat(Y);

train_actual = cell2mat(T_train_rnn);
test_actual = cell2mat(T_test_rnn);

% Plot Results
figure;
plot(test_actual, 'b-', 'LineWidth', 2, 'DisplayName', 'Actual');
hold on;
plot(Y, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
title("RNN - Test Partition: Actual vs Predicted");
legend;
xlabel('Time Step');
ylabel('Value');
grid on;

% Calculate Performance Metrics
test_mse = mean((test_actual - Y).^2);
test_rmse = sqrt(test_mse);
test_mae = mean(abs(test_actual - Y));
test_mean = mean(test_actual);
ss_total = sum((test_actual - test_mean).^2);
ss_residual = sum((test_actual - Y).^2);
r_squared = 1 - (ss_residual / ss_total);

fprintf('\n=== RNN Test Performance ===\n');
fprintf('MSE:  %.4f\n', test_mse);
fprintf('RMSE: %.4f\n', test_rmse);
fprintf('MAE:  %.4f\n', test_mae);
fprintf('R-squared: %.4f\n', r_squared);