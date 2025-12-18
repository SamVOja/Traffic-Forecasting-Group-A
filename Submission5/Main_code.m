close all; clearvars; clc

% Load data
data = readtable('dataset.arff', 'FileType', 'text');

rawTime = data.Var8; 
cleanTime = erase(rawTime, ''''); % Remove all single quotes
data.Var8 = datetime(cleanTime, 'InputFormat', 'yyyy-MM-dd HH:mm:ss'); 
data = sortrows(data, 'Var8'); % Already sorted

target = data.Var9; % traffic volume
t  = data.Var8; % time
temp = data.Var2; % temperature

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

% Remove duplicates and aggregate target
[unique_t, ia, ic] = unique(t);
target_clean = accumarray(ic, target, [], @mean);
temp_clean   = accumarray(ic, temp, [], @mean);  
t_clean = unique_t;

fprintf('Removed %d duplicate timestamps (%d -> %d points)\n', ...
        length(t) - length(t_clean), length(t), length(t_clean));

% Use cleaned data for analysis
target = target_clean;
temp   = temp_clean;   
t      = t_clean;

t_full = (t(1):hours(1):t(end))';
TT  = timetable(t, target, temp);
TTf = retime(TT, t_full, 'fillwithmissing');

t    = TTf.t;
y    = TTf.target;
temp = TTf.temp;

%% Interpolation using seasonality (target + temperature)
missing_y    = isnan(y);
missing_temp = isnan(temp);

% Seasonal components
H = hour(t);      % hour
D = weekday(t);   % weekday
M = month(t);     % month
Y = year(t);      % year

for i = find(missing_y)'
    h   = H(i);
    d   = D(i);
    m   = M(i);
    yrr = Y(i);

    % hour + weekday + month + year
    idx = (H==h) & (D==d) & (M==m) & (Y==yrr) & ~missing_y;
    if any(idx)
        y(i) = mean(y(idx),'omitnan');
        continue
    end

    % hour + weekday + month
    idx = (H==h) & (D==d) & (M==m) & ~missing_y;
    if any(idx)
        y(i) = mean(y(idx),'omitnan');
        continue
    end

    % hour + weekday
    idx = (H==h) & (D==d) & ~missing_y;
    if any(idx)
        y(i) = mean(y(idx),'omitnan');
        continue
    end

    % hour + month
    idx = (H==h) & (M==m) & ~missing_y;
    if any(idx)
        y(i) = mean(y(idx),'omitnan');
        continue
    end

    % hour only
    idx = (H==h) & ~missing_y;
    if any(idx)
        y(i) = mean(y(idx),'omitnan');
        continue
    end

    % global fallback
    y(i) = mean(y(~missing_y),'omitnan');
end

for i = find(missing_temp)'
    h   = H(i);
    d   = D(i);
    m   = M(i);
    yrr = Y(i);

    % hour + weekday + month + year
    idx = (H==h) & (D==d) & (M==m) & (Y==yrr) & ~missing_temp;
    if any(idx)
        temp(i) = mean(temp(idx),'omitnan');
        continue
    end

    % hour + weekday + month
    idx = (H==h) & (D==d) & (M==m) & ~missing_temp;
    if any(idx)
        temp(i) = mean(temp(idx),'omitnan');
        continue
    end

    % hour + weekday
    idx = (H==h) & (D==d) & ~missing_temp;
    if any(idx)
        temp(i) = mean(temp(idx),'omitnan');
        continue
    end

    % hour + month
    idx = (H==h) & (M==m) & ~missing_temp;
    if any(idx)
        temp(i) = mean(temp(idx),'omitnan');
        continue
    end

    % hour only
    idx = (H==h) & ~missing_temp;
    if any(idx)
        temp(i) = mean(temp(idx),'omitnan');
        continue
    end

    % global fallback
    temp(i) = mean(temp(~missing_temp),'omitnan');
end

target    = y;
temp_full = temp;

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

%% Remove Outliers
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

%% Feature Preparation 
hours = hour(t);
day_of_week = weekday(t);

months = month(t);
month_sin = sin(2 * pi * (months - 1) / 12);
month_cos = cos(2 * pi * (months - 1) / 12);

target_lag_1 = [mean(target); target(1:end-1)];      % Previous hour
target_lag_24 = [mean(target)*ones(24,1); target(1:end-24)];  % Previous day same hour
target_lag_168 = [mean(target)*ones(168,1); target(1:end-168)]; % Previous week same hour

features = [target, hours, day_of_week, temp_full, month_sin, month_cos, target_lag_168];
feature_names = {'Traffic', 'Hour', 'DayOfWeek', 'Temperature', 'MonthSin', 'MonthCos', 'LagWeek'};

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

%% Baseline results using a simple Autoregressive Model
run("ARmodel.m"); 

%% RNN
run("RNNmodel.m"); 

%% LSTM 
run("LSTMmodel.m"); 

%% Multivariate Forecasting Transformer
run("MVFTmodel.m"); 

%% Compare models

fprintf('\n=== Model Comparison (Test Performance) ===\n');
fprintf('Model        | RMSE    | MAE     | R-squared\n');
fprintf('------------|---------|---------|----------\n');
fprintf('AR          | %.4f | %.4f | %.4f\n', test_rmse_ar, test_mae_ar, r_squared_ar);
fprintf('RNN         | %.4f | %.4f | %.4f\n', test_rmse_rnn, test_mae_rnn, r_squared_rnn);
fprintf('LSTM        | %.4f | %.4f | %.4f\n', test_rmse_lstm, test_mae_lstm, r_squared_lstm);
fprintf('Transformer | %.4f | %.4f | %.4f\n', test_rmse_transformer, test_mae_transformer, test_r_squared_transformer);