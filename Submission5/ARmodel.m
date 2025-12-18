% AR MODEL 

y_train = train_target;
y_test = test_target;

ar_order = 24; % 1 day
idata_train = iddata(y_train, []);
sys = ar(idata_train, ar_order); % train model

A = sys.A;  % AR coefficients

% Ensure temp_window has correct length
if length(A)-1 ~= ar_order
    % Use actual number of coefficients from the model
    actual_order = length(A) - 1;
    ar_order = actual_order;
end

% Make one-step ahead predictions on training data (validation)
YCalPred = zeros(length(y_train)-ar_order, 1);
for i = ar_order:length(y_train)-1
    prev_vals = y_train(i-ar_order+1:i);

    % Predict next value
    YCalPred(i-ar_order+1) = -A(2:end) * flipud(prev_vals(:)); % column vector
end

% Make one-step ahead predictions on test data 
Y_full = zeros(length(y_test), 1);
window = y_train(end-ar_order+1:end); % Initialize with last training values
for i = 1:length(y_test)
    % Predict next value 
    Y_full(i) = -A(2:end) * flipud(window(:));
    
    % Use actual test value for next prediction
    window = [window(2:end); y_test(i)];
end

% Align actual values
train_actual = y_train(ar_order+1:end);
test_actual = y_test;

% Full plot
figure;
plot(test_actual, 'b-', 'LineWidth', 2, 'DisplayName', 'Actual');
hold on;
plot(Y_full, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
title(sprintf('AR(%d) - Test Partition: Actual vs Predicted', ar_order));
legend;
xlabel('Time Step');
ylabel('Traffic Volume');
grid on;

% Zoom in on last week
figure;
hours_per_week = 168;
last_week_start = max(1, length(test_actual) - hours_per_week + 1);
last_week_end = length(test_actual);
plot(1:168, test_actual(last_week_start:last_week_end), 'b-', 'LineWidth', 2);
hold on;
plot(1:168, Y_full(last_week_start:last_week_end), 'r--', 'LineWidth', 1.5);
title(sprintf('AR(%d) - Last Week', ar_order));
xlabel('Hour');
ylabel('Traffic Volume');
grid on;
legend('Actual', 'Predicted');

% Calculate Performance Metrics for Model 1
test_mse_ar = mean((test_actual - Y_full).^2);
test_rmse_ar = sqrt(test_mse_ar);
test_mae_ar = mean(abs(test_actual - Y_full));
test_mean = mean(test_actual);
ss_total = sum((test_actual - test_mean).^2);
ss_residual = sum((test_actual - Y_full).^2);
r_squared_ar = 1 - (ss_residual / ss_total);

fprintf('\n=== Baseline AR(24) Model Test Performance ===\n');
fprintf('MSE:  %.4f\n', test_mse_ar);
fprintf('RMSE: %.4f\n', test_rmse_ar);
fprintf('MAE:  %.4f\n', test_mae_ar);
fprintf('R-squared: %.4f\n', r_squared_ar);