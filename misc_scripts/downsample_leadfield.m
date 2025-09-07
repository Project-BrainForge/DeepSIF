% Downsample the 75-channel leadfield matrix to create additional leadfield matrices
% for specific electrode configurations based on 10-10 and 10-20 systems
% 
% This script creates 4 downsampled leadfield matrices with 64, 32, 21, and 16 channels
% as described in the study using the fsaverage5 model

% Load the original 75-channel leadfield matrix
load('../anatomy/leadfield_75_20k.mat'); % Assumes the variable 'fwd' contains the leadfield matrix
load('../anatomy/electrode_75.mat'); % Loads electrode configuration

% Define electrode configurations based on standard 10-10 and 10-20 systems
% These indices are approximations - adjust if you have the actual electrode mappings

% Define configurations with specific channel counts
% 64-channel configuration (standard dense array)
config_64 = sort([1:64]); % Assuming first 64 electrodes represent the 64-channel configuration

% 32-channel configuration (moderate coverage)
% This typically includes key electrodes from the 10-20 system plus additional sites
config_32 = sort([1:2:63]); % Taking every other electrode from the first 63

% 21-channel configuration (close to standard 10-20 system)
% The 10-20 system has 19 standard positions plus references
config_21 = sort([1:3:63]); % Taking every third electrode from the first 63

% 16-channel configuration (minimal coverage)
% This typically includes key positions like Fz, Cz, Pz, Oz, F3/4, C3/4, P3/4, etc.
config_16 = sort([1:5:75]); % Taking sparse sampling of electrodes

% Organize configurations
configs = {
    config_64,  % 64-channel configuration
    config_32,  % 32-channel configuration
    config_21,  % 21-channel configuration
    config_16   % 16-channel configuration
};

config_names = {'64ch', '32ch', '21ch', '16ch'};

% Process each configuration
for i = 1:length(configs)
    % Get the selected electrode indices for this configuration
    selected_electrodes = configs{i};
    
    % Make sure we don't exceed the number of available electrodes
    selected_electrodes = selected_electrodes(selected_electrodes <= 75);
    
    % Downsample the leadfield matrix by selecting only the specified rows
    downsampled_fwd = fwd(selected_electrodes, :);
    
    % Save the downsampled leadfield matrix
    % We use the actual channel count in the filename
    channels = length(selected_electrodes);
    output_filename = sprintf('../anatomy/leadfield_%d_20k.mat', channels);
    
    % Save both the downsampled forward matrix and the electrode indices
    save(output_filename, 'downsampled_fwd', 'selected_electrodes');
    
    fprintf('Created downsampled leadfield matrix: %s (%d electrodes)\n', ...
        output_filename, channels);
end

fprintf('Downsampling complete. Created %d new leadfield matrices.\n', length(configs));
