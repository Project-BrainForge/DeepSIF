% Load data  
load('./fs_cortex_20k.mat');
load('./fs_cortex_20k_region_mapping.mat');
load('/MATLAB Drive/transformer_test_debug_best.pth.mat');

% Extract temporal data: (350 × 994)
temporal_data = squeeze(all_out(10, :, :));
% Find the time point with maximum global activity
global_activity = sum(abs(temporal_data), 2);  % Sum across all regions
[~, peak_time] = max(global_activity);

fprintf('Peak activity occurs at time point %d\n', peak_time);

% Visualize the peak activity time point
reconstruction_data = zeros(1, size(pos, 1));
for i = 1:994
    region_vertices = (rm == (i-1));
    reconstruction_data(region_vertices) = temporal_data(peak_time, i);
end

visualize_result(pos, tri, reconstruction_data, ...
    'FaceAlpha', 0.8, ...
    'thre', 0.2, ...
    'titles', {{sprintf('Peak Activity (t=%d)', peak_time)}});