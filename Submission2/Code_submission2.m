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

%% Time-series data partitioning plan
% 70% training, 30% testing, chronological
n = height(data);
train_end = round(0.7 * n);

trainData = data(1:train_end, :);
testData  = data(train_end+1:end, :);