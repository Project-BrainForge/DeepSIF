% Visualize VEP Results
% This script visualizes the all_out variable (3 x 500 x 994)
clear; close all;

% Load data
data = load('source/VEP/rnn_test_1_.pth.tar.mat');
all_out = data.all_out; % 3 x 500 x 994

% Load cortex anatomy
anat = load('anatomy/fs_cortex_20k.mat');
pos = anat.pos;
tri = anat.tri;

% Select subset of vertices (994 out of 20k)
% You may need to adjust this based on your region mapping
vertex_indices = 1:994; % Adjust if you have specific mapping

%% Visualization Options

% 1. Visualize specific time points for all 3 samples
time_points = [100, 250, 400]; % Select interesting time points
figure('Position', [100, 100, 1400, 800]);

for sample = 1:3
    for t_idx = 1:length(time_points)
        t = time_points(t_idx);
        subplot(3, 3, (sample-1)*3 + t_idx);
        
        % Create full-size activation map
        value = zeros(1, size(pos, 1));
        value(vertex_indices) = squeeze(all_out(sample, t, :));
        
        % Add path to visualize_result
        addpath('misc_scripts');
        visualize_result(pos, tri, value, 'row', 1, 'col', 1, ...
                        'thre', 0.3, 'new_fig', 0, ...
                        'titles', {sprintf('Sample %d, Time %d', sample, t)});
        title(sprintf('Sample %d, t=%d ms', sample, t));
    end
end
sgtitle('Brain Activation at Different Time Points');

%% 2. Time series visualization for peak vertices
figure('Position', [100, 100, 1200, 600]);
for sample = 1:3
    subplot(1, 3, sample);
    
    % Get data for this sample
    data_sample = squeeze(all_out(sample, :, :)); % 500 x 994
    
    % Find top 5 most active vertices
    max_activity = max(abs(data_sample), [], 1);
    [~, top_idx] = sort(max_activity, 'descend');
    top_vertices = top_idx(1:5);
    
    % Plot time series
    hold on;
    for v = 1:5
        plot(1:500, data_sample(:, top_vertices(v)), 'LineWidth', 1.5, ...
             'DisplayName', sprintf('Vertex %d', top_vertices(v)));
    end
    xlabel('Time (samples)');
    ylabel('Activation');
    title(sprintf('Sample %d - Top 5 Vertices', sample));
    legend('Location', 'best');
    grid on;
end
sgtitle('Time Series of Most Active Vertices');

%% 3. Heatmap visualization
figure('Position', [100, 100, 1400, 400]);
for sample = 1:3
    subplot(1, 3, sample);
    data_sample = squeeze(all_out(sample, :, :)); % 500 x 994
    
    imagesc(data_sample');
    colorbar;
    xlabel('Time (samples)');
    ylabel('Vertex Index');
    title(sprintf('Sample %d - Spatiotemporal Pattern', sample));
    colormap('jet');
end
sgtitle('Spatiotemporal Heatmaps');

%% 4. Statistical summary across samples
figure('Position', [100, 100, 1200, 500]);

% Average across 3 samples
mean_activation = squeeze(mean(all_out, 1)); % 500 x 994

subplot(1, 2, 1);
imagesc(mean_activation');
colorbar;
xlabel('Time (samples)');
ylabel('Vertex Index');
title('Mean Activation Across 3 Samples');
colormap('jet');

subplot(1, 2, 2);
std_activation = squeeze(std(all_out, 0, 1)); % 500 x 994
imagesc(std_activation');
colorbar;
xlabel('Time (samples)');
ylabel('Vertex Index');
title('Std Dev Across 3 Samples');
colormap('hot');

fprintf('Visualization complete!\n');
fprintf('Data shape: [%d samples, %d timepoints, %d vertices]\n', ...
        size(all_out, 1), size(all_out, 2), size(all_out, 3));
fprintf('Value range: [%.4f, %.4f]\n', min(all_out(:)), max(all_out(:)));
