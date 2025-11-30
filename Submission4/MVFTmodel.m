% Multivariate Forecasting Transformer Model

% Parameters for Transformer
sequenceLength = 24; % Use 1 day as sequence length (same as LSTM for comparison)
d_model = 128; 
numHeads = 8;
numEncoderLayers = 3; 
ffnHiddenSize = 256;
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
    
    % Feature embedding to higher dimension
    fullyConnectedLayer(d_model, 'Name', 'feature_embedding')
    
    % Self-attention mechanism
    selfAttentionLayer(numHeads, d_model, 'Name', 'attention1')
    
    % Layer normalization
    layerNormalizationLayer('Name', 'layernorm1')
    
    % Feed-forward network
    fullyConnectedLayer(ffnHiddenSize, 'Name', 'ffn_fc1')
    reluLayer('Name', 'ffn_relu')
    dropoutLayer(dropoutProb, 'Name', 'ffn_dropout1')
    fullyConnectedLayer(d_model, 'Name', 'ffn_fc2')
    
    % Second layer normalization
    layerNormalizationLayer('Name', 'layernorm2')
    
    % Global pooling across time dimension
    globalAveragePooling1dLayer('Name', 'global_avg_pool')
    
    % Output layers
    fullyConnectedLayer(32, 'Name', 'output_fc1')
    reluLayer('Name', 'output_relu')
    dropoutLayer(0.2, 'Name', 'output_dropout')
    fullyConnectedLayer(1, 'Name', 'output_fc2')
    regressionLayer('Name', 'output')
];

% Network architecture
figure;
plot(layerGraph(layers));
title('Transformer Architecture');

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.5, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XTest_transformer, YTest_transformer}, ...
    'ValidationFrequency', 100, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the Transformer model
fprintf('Training Transformer model...\n');
netTransformer = trainNetwork(XTrain_transformer, YTrain_transformer, layers, options);

% Make predictions
fprintf('Making predictions with Transformer...\n');
YTrainPred_transformer = predict(netTransformer, XTrain_transformer);
YTestPred_transformer = predict(netTransformer, XTest_transformer);

% Calculate Performance Metrics
test_mse_transformer = mean((YTest_transformer - YTestPred_transformer).^2);
test_rmse_transformer = sqrt(test_mse_transformer);
test_mae_transformer = mean(abs(YTest_transformer - YTestPred_transformer));

test_mean = mean(YTest_transformer);
test_ss_total = sum((YTest_transformer - test_mean).^2);
test_ss_residual = sum((YTest_transformer - YTestPred_transformer).^2);
test_r_squared = 1 - (test_ss_residual / test_ss_total);

fprintf('\n=== Transformer Test Performance ===\n');
fprintf('MSE:  %.4f\n', test_mse_transformer);
fprintf('RMSE: %.4f\n', test_rmse_transformer);
fprintf('MAE:  %.4f\n', test_mae_transformer);
fprintf('R-squared: %.4f\n', test_r_squared);

figure;
plot(YTest_transformer, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Actual');
hold on;
plot(YTestPred_transformer, 'r--', 'LineWidth', 1, 'DisplayName', 'Transformer Predicted');
title('Transformer - Test Partition: Actual vs Predicted');
legend;
xlabel('Time Step');
ylabel('Traffic Volume');
grid on;