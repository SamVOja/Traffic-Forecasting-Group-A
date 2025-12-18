% Multivariate Forecasting Transformer Model

% Parameters for Transformer
sequenceLength = 24; % Use 1 day as sequence length 
d_model = 64;
numHeads = 4;

numEncoderLayers = 3; 
ffnHiddenSize = 128;

dropoutProb = 0.1;
maxPosition = 1000;

fprintf('Implementing Multivariate Forecasting Transformer...\n');

XTrain_transformer = {};
YTrain_transformer = [];
for i = 1:length(train_target) - sequenceLength
    XTrain_transformer{end+1} = train_features(i:i+sequenceLength-1, :)';
    YTrain_transformer(end+1) = train_target(i+sequenceLength);
end
XTrain_transformer = XTrain_transformer';
YTrain_transformer = YTrain_transformer';

XTest_transformer = {};
YTest_transformer = [];
for i = 1:length(test_target) - sequenceLength
    XTest_transformer{end+1} = test_features(i:i+sequenceLength-1, :)';
    YTest_transformer(end+1) = test_target(i+sequenceLength);
end
XTest_transformer = XTest_transformer';
YTest_transformer = YTest_transformer';

layers = [
    sequenceInputLayer(numFeatures, 'Name', 'input')

    fullyConnectedLayer(64, 'Name', 'feature_embedding')
    layerNormalizationLayer('Name','embed_norm')

    fullyConnectedLayer(64, 'Name','attn_proj')
    tanhLayer('Name','attn_act')
    layerNormalizationLayer('Name','attn_norm')

    indexing1dLayer("last", 'Name', 'last_step')

    fullyConnectedLayer(32, 'Name', 'output_fc1')
    reluLayer('Name', 'output_relu')
    fullyConnectedLayer(1, 'Name', 'output_fc2')

    regressionLayer('Name', 'output')
];


% Network architecture
figure;
plot(layerGraph(layers));
title('Transformer Architecture');

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 16, ...
    'InitialLearnRate', 0.0002, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 15, ...
    'LearnRateDropFactor', 0.5, ...
    'GradientThreshold', 0.5, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest_transformer, YTest_transformer}, ...
    'ValidationFrequency', 200, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

fprintf('Training Transformer\n');
netTransformer = trainNetwork(XTrain_transformer, YTrain_transformer, layers, options);

YTrainPred_transformer = predict(netTransformer, XTrain_transformer);
YTestPred_transformer = predict(netTransformer, XTest_transformer);

figure;
plot(YTest_transformer, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(YTestPred_transformer, 'r--', 'LineWidth', 1, 'DisplayName', 'Transformer Predicted');
title('Transformer - Test Partition: Actual vs Predicted');
legend;
xlabel('Time Step');
ylabel('Traffic Volume');
grid on;

% Zoom in on last week
figure;
hours_per_week = 168;
last_week_start = max(1, length(YTest_transformer) - hours_per_week + 1);
last_week_end = length(YTest_transformer);
plot(1:168, YTest_transformer(last_week_start:last_week_end), 'b-', 'LineWidth', 2);
hold on;
plot(1:168, YTestPred_transformer(last_week_start:last_week_end), 'r--', 'LineWidth', 1.5);
title(sprintf('MVFT - Last Week'));
xlabel('Hour');
ylabel('Traffic Volume');
grid on;
legend('Actual', 'Predicted');

% Calculate Performance Metrics
test_mse_transformer = mean((YTest_transformer - YTestPred_transformer).^2);
test_rmse_transformer = sqrt(test_mse_transformer);
test_mae_transformer = mean(abs(YTest_transformer - YTestPred_transformer));

test_mean = mean(YTest_transformer);
test_ss_total = sum((YTest_transformer - test_mean).^2);
test_ss_residual = sum((YTest_transformer - YTestPred_transformer).^2);
test_r_squared_transformer = 1 - (test_ss_residual / test_ss_total);

fprintf('\n=== Transformer Test Performance ===\n');
fprintf('MSE:  %.4f\n', test_mse_transformer);
fprintf('RMSE: %.4f\n', test_rmse_transformer);
fprintf('MAE:  %.4f\n', test_mae_transformer);
fprintf('R-squared: %.4f\n', test_r_squared_transformer);

