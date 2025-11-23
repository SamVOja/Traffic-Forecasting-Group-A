close all; clearvars; clc

% Load data
data = readtable('dataset.arff', 'FileType', 'text');

rawTime = data.Var8; 
cleanTime = erase(rawTime, ''''); % Remove all single quotes
data.Var8 = datetime(cleanTime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss'); 
data = sortrows(data, 'Var8'); % Already sorted

target = data.Var9; % traffic volume
t  = data.Var8;

%% Time-series visualization (before cleaning the data)
figure;
plot(t, target);
title('Traffic Volume Over Time (Data gaps highlighted with red)');
xlabel('Time'); 
ylabel('Traffic Volume');
grid on;

if isdatetime(t)
    % Find gaps longer than 1 day
    time_diff = diff(t);
    large_gaps = find(time_diff > days(1));
    
    hold on;
    for i = 1:length(large_gaps)
        gap_start = t(large_gaps(i));
        gap_end = t(large_gaps(i)+1);
        ylims = ylim;
        patch([gap_start, gap_end, gap_end, gap_start], ...
              [ylims(1), ylims(1), ylims(2), ylims(2)], ...
              [1, 0, 0], 'EdgeColor', 'r');
    end
end

%% Clean data

[unique_t, ia, ic] = unique(t);
target_clean = accumarray(ic, target, [], @mean);
t_clean = unique_t;
fprintf('Removed %d duplicate timestamps (%d -> %d points)\n', ...
        length(t) - length(t_clean), length(t), length(t_clean));

% Use cleaned data for analysis
target = target_clean;
t = t_clean;

%% Interpolation using seasonality

t_full = (t(1):hours(1):t(end))';
TT = timetable(t, target);
TTf = retime(TT, t_full, 'fillwithmissing');
t = TTf.t;
y = TTf.target;

missing = isnan(y);

% Seasonal components
H = hour(t);      % hour 
D = weekday(t);   % weekday
M = month(t);     % month 
Y = year(t);      % year

for i = find(missing)'
    h = H(i);
    d = D(i);
    m = M(i);
    yrr = Y(i);

    % hour + weekday + month + year
    idx = (H==h) & (D==d) & (M==m) & (Y==yrr) & ~missing;
    if any(idx)
        y(i) = mean(y(idx), 'omitnan');
        continue
    end

    % hour + weekday + month 
    idx = (H==h) & (D==d) & (M==m) & ~missing;
    if any(idx)
        y(i) = mean(y(idx), 'omitnan');
        continue
    end

    % hour + weekday
    idx = (H==h) & (D==d) & ~missing;
    if any(idx)
        y(i) = mean(y(idx), 'omitnan');
        continue
    end

    % hour + month
    idx = (H==h) & (M==m) & ~missing;
    if any(idx)
        y(i) = mean(y(idx), 'omitnan');
        continue
    end

    % hour
    idx = (H==h) & ~missing;
    if any(idx)
        y(i) = mean(y(idx), 'omitnan');
        continue
    end
    y(i) = mean(y(~missing));
end
target = y;

%% Time-series decomposition (trend, seasonality, residual)
tt = timetable(t, target);
window = min(24, floor(length(target)/10)); % Adaptive window size
trend = movmean(target, window);
detrended = target - trend;

seasonal_daily   = zeros(size(target));
seasonal_weekly  = zeros(size(target));
seasonal_monthly = zeros(size(target));

% daily seasonality
for h = 0:23
    idx = hour(t) == h;
    seasonal_daily(idx) = mean(detrended(idx), 'omitnan');
end

% weekly seasonality
for wd = 1:7
    idx = weekday(t) == wd;
    seasonal_weekly(idx) = mean(detrended(idx), 'omitnan');
end

% monthly seasonality
for m = 1:12
    idx = month(t) == m;
    seasonal_monthly(idx) = mean(detrended(idx), 'omitnan');
end

seasonal_total = seasonal_daily + seasonal_weekly + seasonal_monthly;
residual = detrended - seasonal_total;

figure;
subplot(4,1,1); plot(t, target); title('Original'); ylabel('Volume');
subplot(4,1,2); plot(t, trend); title('Trend'); ylabel('Volume');
subplot(4,1,3); plot(t, seasonal_monthly); title('Monthly Seasonality'); ylabel('Volume');
subplot(4,1,4); plot(t, residual); title('Residual'); ylabel('Volume');
xlabel('Time');
sgtitle('Trend, Seasonality, Residual');

%% Detect outliers in residuals
mad_res = mad(residual,1);             % Median absolute deviation
median_res = median(residual);         % Median
outlier_idx = abs(residual - median_res) > 8*mad_res;  

figure;
plot(t, residual, 'b'); hold on;
plot(t(outlier_idx), residual(outlier_idx), 'ro', 'MarkerSize',6);
title('Residuals with Outliers Highlighted');
xlabel('Time'); ylabel('Residual');
legend('Residual','Outlier');
grid on;

target(outlier_idx) = NaN;
target = fillmissing(target, 'linear'); % interpolate linearly

%% Autocorrelation analysis
figure;
autocorr(target, 'NumLags', 200);
title('Autocorrelation of Traffic Volume');

%% Traffic volume by hour
hours = hour(t);  

figure;
scatter(hours, target, 10, 'filled');
title('Traffic Volume by Hour of Day');
xlabel('Hour of Day');
ylabel('Traffic Volume');
grid on;

%% Summary statistics
fprintf('Traffic Volume Statistics:\n');
fprintf('Mean: %.2f\n', mean(target));
fprintf('Median: %.2f\n', median(target));
fprintf('Standard Deviation: %.2f\n', std(target));
fprintf('Minimum: %.2f\n', min(target));
fprintf('Maximum: %.2f\n', max(target));

%% Baseline results using a simple Autoregressive Model
y = target;     % provides input to the mlx script

run("AutoregressiveModel.mlx"); 

%% Feature Preparation %TODO move to earlier
hours = hour(t);
day_of_week = weekday(t);
temp = data.Var2;

%TODO also seasonally interpolate temperature
temp_clean = accumarray(ic, temp, [], @mean);
temp_full = interp1(t_clean, temp_clean, t_full, 'linear');
temp_full = fillmissing(temp_full, 'linear');

features = [target, hours, day_of_week, temp_full];
feature_names = {'Traffic', 'Hour', 'DayOfWeek', 'Temperature'};
numFeatures = size(features, 2); 
fprintf('Features: %s\n', strjoin(feature_names, ', '));

%% Normalize features 
feature_means = mean(features, 1, 'omitnan');
feature_stds = std(features, 0, 1, 'omitnan');

feature_stds(feature_stds == 0) = 1; % Avoid division by zero
features_normalized = (features - feature_means) ./ feature_stds;

% Store normalization parameters 
norm_params.means = feature_means;
norm_params.stds = feature_stds;
norm_params.feature_names = feature_names;

%% Time-series data partitioning 
n = length(target);
train_end = round(0.7 * n);

% Split normalized features
train_features = features_normalized(1:train_end, :);
test_features = features_normalized(train_end+1:end, :);

% Split target 
train_target = target(1:train_end);
test_target = target(train_end+1:end);

% Split time vector
train_time = t(1:train_end);
test_time = t(train_end+1:end);

fprintf('Data partitioning:\n');
fprintf('Training set: %d samples (%.1f%%)\n', length(train_target), 100*length(train_target)/n);
fprintf('Testing set:  %d samples (%.1f%%)\n', length(test_target), 100*length(test_target)/n);

%% RNN
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

%% LSTM 
numTimeSteps = 24; % 1 day lookback window

% Training sequences for LSTM
XTrain_lstm = {};
YTrain_lstm = [];
for i = 1:length(train_target) - numTimeSteps
    XTrain_lstm{end+1} = train_features(i:i+numTimeSteps-1, :)';
    YTrain_lstm(end+1) = train_target(i+numTimeSteps);
end
XTrain_lstm = XTrain_lstm';
YTrain_lstm = YTrain_lstm';

% Testing sequences for LSTM
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
    lstmLayer(50, 'OutputMode', 'last')              
    dropoutLayer(0.2)
    fullyConnectedLayer(25)
    reluLayer()
    fullyConnectedLayer(1)
    regressionLayer()];

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...                         
    'InitialLearnRate', 0.001, ...                  
    'GradientThreshold', 1, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 25, ...
    'LearnRateDropFactor', 0.5, ...
    'ValidationData', {XTest_lstm, YTest_lstm}, ... 
    'ValidationFrequency', 10, ...
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
