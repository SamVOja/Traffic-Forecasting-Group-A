% RNN MODEL

% Single-step prediction 
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

N = numel(X_train_rnn);

net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:floor(0.7*N);
net.divideParam.valInd   = floor(0.7*N)+1 : floor(0.85*N);
net.divideParam.testInd  = floor(0.85*N)+1 : N;

[trainedNet, tr] = train(net, X_train_rnn, T_train_rnn);

% layrecnet does not plot the training itself.
if isfield(tr,'perf') && ~isempty(tr.perf)
    figure;
    plot(tr.perf,'LineWidth',2);
    xlabel('Epoch');
    ylabel('Training MSE');
    grid on;
end

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

% Zoom in on last week
figure;
hours_per_week = 168;
last_week_start = max(1, length(test_actual) - hours_per_week + 1);
last_week_end = length(test_actual);
plot(1:168, test_actual(last_week_start:last_week_end), 'b-', 'LineWidth', 2);
hold on;
plot(1:168, Y(last_week_start:last_week_end), 'r--', 'LineWidth', 1.5);
title('RNN - Last Week');
xlabel('Hour');
ylabel('Traffic Volume');
grid on;
legend('Actual', 'Predicted');

% Calculate Performance Metrics
test_mse_rnn = mean((test_actual - Y).^2);
test_rmse_rnn = sqrt(test_mse_rnn);
test_mae_rnn = mean(abs(test_actual - Y));
test_mean = mean(test_actual);
ss_total = sum((test_actual - test_mean).^2);
ss_residual = sum((test_actual - Y).^2);
r_squared_rnn = 1 - (ss_residual / ss_total);

fprintf('\n=== RNN Test Performance ===\n');
fprintf('MSE:  %.4f\n', test_mse_rnn);
fprintf('RMSE: %.4f\n', test_rmse_rnn);
fprintf('MAE:  %.4f\n', test_mae_rnn);
fprintf('R-squared: %.4f\n', r_squared_rnn);