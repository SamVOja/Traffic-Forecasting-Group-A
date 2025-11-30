% Use AR model trained on three weeks of data
idata = iddata(y(end-168*3:end-168), []);  % exclude last week
sys = ar(idata, 168);

K = 168;
p = forecast(sys, idata, K);  
y_AR = p.OutputData;  % forecast for the last week

% Indices
N = numel(y);
idx_prev = (N-335):(N-168);   % previous week (168 points)
idx_last = (N-167):N;         % last week (168 points)

% Vectors
t_forecast_AR = t(idx_last);  % align forecast with last week
y_AR_plot = [y(idx_prev(end)); y_AR];
t_AR_plot = [t(idx_prev(end)); t_forecast_AR];

% Plot
figure;
plot(t(idx_prev), y(idx_prev), 'k', 'LineWidth',1.5); hold on; % previous week
plot(t(idx_last), y(idx_last), 'b', 'LineWidth',1.5);          % actual last week
plot(t_AR_plot, y_AR_plot, 'r', 'LineWidth',1.2);              % forecast

title('Previous Week, Last Week, and AR Forecast for Last Week');
xlabel('Time'); ylabel('Traffic Volume');
legend('Previous Week','Actual Last Week','Forecast Last Week');
grid on;

