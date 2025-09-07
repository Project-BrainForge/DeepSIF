% Evaluate estimated sources based on saved resource after performing otsu
% threshold.
clear
dataset_name = '_sample_source2';
fname = '_sample_source2';
model_id = '75';
gt = load(['../source/Simulation/test' dataset_name '.mat']);             % ground truth
num_source = size(gt.selected_region, 2);
load(['../model_result/' model_id '_the_model/model_best.pth.tar_preds_test' fname '.mat']);

load('../anatomy/dis_matrix_fs_20k.mat')
all_dis = raw_dis_matrix;
% Variabled loaded:
% FROM source
% gt.selected_region   : array; ground truth source region, start from 0, num_examples * num_source * MAX_SIZE 
%                    (MAX_SIZE=70, the number of cortical regions in each 
%                    example is different, padd with 15213 to size 70
% FROM model_result
% all_regions          : cell; DeepSIF reconstructed source region, start from 0, num_examples * 1
% all_out              : cell; activity in DeepSIF reconstructed source region; num_examples * 1
% all_num              : array: ground truth activity; num_examples * num_source * num_time
% FROM anatomy
% all_dis              : distance between cortical source regions

%%
precision = nan(length(all_out), num_source);
recall = nan(length(all_out), num_source);
le = nan(length(all_out), num_source);
all_corr = nan(length(all_out), num_source);

recon_regions = cell(length(all_out), num_source);
recon_activity = cell(length(all_out), num_source);

for i= 1:length(all_out)
    
    % gather all source regions, remove padded variable.
    all_label = reshape(squeeze(gt.selected_region(i,:,:))',[],num_source);
    num_region_per_source = sum(~myisnan(all_label),1);
    all_label(myisnan(all_label)) = [];
    % recon regions
    % Handle different formats that might be saved by Python
    try
        % Try to access as a cell array first
        current_regions = all_regions{i};
    catch
        % If not a cell array, try accessing as a struct with numeric field names
        try
            % Convert i to string and use as field name
            field_name = ['x', num2str(i-1)]; % Python is 0-indexed, MATLAB is 1-indexed
            current_regions = all_regions.(field_name);
        catch
            % If that fails too, try direct indexing if it's a regular array
            try
                current_regions = all_regions(i,:);
            catch
                % Last resort: try to get the i-th row if it's stored differently
                try
                    current_regions = all_regions(:,i);
                catch
                    % Give up and skip this iteration
                    warning(['Could not access all_regions for index ' num2str(i)]);
                    continue;
                end
            end
        end
    end
    
    if isempty(current_regions)
        continue
    end
    
    % assign the each recon source region to its closest source patch
    [~, min_ind] = min(all_dis(all_label+1,current_regions+1),[],1);
    mapping = []; % size: 1 * total_ground_truth_source_regions
    for ii=1:length(num_region_per_source)
        mapping = [mapping ii*ones(1,num_region_per_source(ii))];
    end
    source_id = mapping(min_ind);
    
    % calculate metrics
    for k=1:max(source_id)
        recon = current_regions(source_id==k);
        lb =  squeeze(gt.selected_region(i,k,:));
        lb(myisnan(lb)) = [];
        if ~isempty(recon)
            recon_regions{i,k} = recon; 
            % Handle different formats for all_out similarly
            try
                recon_activity{i,k} = all_out{i}(source_id==k,:);
            catch
                try
                    field_name = ['x', num2str(i-1)];
                    recon_activity{i,k} = all_out.(field_name)(source_id==k,:);
                catch
                    try
                        recon_activity{i,k} = all_out(i, source_id==k,:);
                    catch
                        warning(['Could not access all_out for index ' num2str(i)]);
                        recon_activity{i,k} = [];
                    end
                end
            end
            
            interc = intersect(recon,lb);
            precision(i,k) = length(interc)/length(recon);
            recall(i,k) = length(interc)/length(lb);
            all_corr(i,k) = corr(mean(all_out{i}(source_id==k,:),1)',squeeze(all_nmm(i,k,:)));
            le(i,k) = mean(min(all_dis(recon+1,lb+1),[],2));
        end
    end    
        
end
% save and display results
s(1) = mean(precision(:), 'omitnan');s(2) = mean(recall(:),'omitnan');s(3) = mean(all_corr(:),'omitnan');s(4) = mean(le(:),'omitnan');s
save(['../model_result/' model_id '_the_model/recon' fname '.mat'],'precision','recall','le','all_corr','recon_regions','recon_activity')
%%
function y = myisnan(x)
    y = abs(x-15213)<1e-6;
end



