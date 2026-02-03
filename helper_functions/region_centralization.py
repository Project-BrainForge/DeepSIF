"""Functions for centralizing brain regions based on activation and labels"""

import numpy as np
import scipy.io as sio
from typing import Tuple, Dict, List, Union
from .region_analysis import get_top_activated_regions, get_top_activated_regions_from_data
from .label_extraction import extract_labels, extract_field


def centralize_regions_by_label(
    mat_file_path: str,
    n_regions: int = 70,
    data_key: str = None,
    label_key: str = 'label',
    aggregation_method: str = 'mean'
) -> Dict[str, Union[np.ndarray, Dict]]:
    """
    Extract top activated regions and centralize them based on their labels.
    Regions with the same label are grouped together, with the highest 
    activated region in each label group as the center.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the .mat file containing brain activation data and labels
    n_regions : int, optional (default=70)
        Number of top activated regions to extract
    data_key : str, optional
        Key name for activation data in mat file
    label_key : str, optional (default='label')
        Key name for label data in mat file
    aggregation_method : str, optional (default='mean')
        Method to aggregate activation across time
    
    Returns:
    --------
    result : Dict
        Dictionary containing:
        - 'top_region_indices': Indices of top N activated regions
        - 'top_activation_values': Activation values for each region
        - 'region_labels': Label for each top region
        - 'centralized_groups': Dict mapping each label to its regions
        - 'center_regions': Dict mapping each label to its center (max activated) region
        - 'sorted_regions': Regions sorted by label groups with centers first
    
    Example:
    --------
    >>> result = centralize_regions_by_label('data.mat', n_regions=70)
    >>> print(f"Found {len(result['centralized_groups'])} label groups")
    """
    # Step 1: Get top activated regions
    print(f"Step 1: Extracting top {n_regions} activated regions...")
    top_indices, top_activations = get_top_activated_regions(
        mat_file_path, 
        n_regions=n_regions,
        data_key=data_key,
        aggregation_method=aggregation_method
    )
    
    # Step 2: Extract labels for all regions
    print(f"\nStep 2: Extracting labels...")
    all_labels = extract_labels(mat_file_path, label_key)
    
    # Get labels for the top activated regions
    region_labels = all_labels[top_indices]
    
    # Step 3: Group regions by labels
    print(f"\nStep 3: Grouping regions by labels...")
    label_groups = {}
    
    for i, (idx, activation, label) in enumerate(zip(top_indices, top_activations, region_labels)):
        label_str = str(label) if not isinstance(label, str) else label
        
        if label_str not in label_groups:
            label_groups[label_str] = {
                'indices': [],
                'activations': [],
                'original_positions': []
            }
        
        label_groups[label_str]['indices'].append(idx)
        label_groups[label_str]['activations'].append(activation)
        label_groups[label_str]['original_positions'].append(i)
    
    # Step 4: Find center (max activated) region for each label group
    print(f"\nStep 4: Finding center regions for each label group...")
    center_regions = {}
    
    for label, group_data in label_groups.items():
        activations = np.array(group_data['activations'])
        abs_activations = np.abs(activations)
        
        # Find index of maximum activation within this label group
        max_idx = np.argmax(abs_activations)
        center_regions[label] = {
            'region_index': group_data['indices'][max_idx],
            'activation': group_data['activations'][max_idx],
            'position_in_top': group_data['original_positions'][max_idx]
        }
    
    # Step 5: Sort regions by label groups with centers first
    print(f"\nStep 5: Centralizing and sorting regions...")
    sorted_regions = []
    sorted_activations = []
    sorted_labels = []
    sorted_info = []
    
    for label in sorted(label_groups.keys()):
        group_data = label_groups[label]
        center_info = center_regions[label]
        
        # Convert to numpy arrays for easier manipulation
        indices = np.array(group_data['indices'])
        activations = np.array(group_data['activations'])
        
        # Find the center region position
        center_pos = np.where(indices == center_info['region_index'])[0][0]
        
        # Reorder: center first, then others sorted by activation
        abs_activations = np.abs(activations)
        sorted_positions = np.argsort(abs_activations)[::-1]
        
        # Add to sorted list (center is already first due to sorting by activation)
        for pos in sorted_positions:
            sorted_regions.append(indices[pos])
            sorted_activations.append(activations[pos])
            sorted_labels.append(label)
            is_center = (indices[pos] == center_info['region_index'])
            sorted_info.append({
                'region_index': indices[pos],
                'activation': activations[pos],
                'label': label,
                'is_center': is_center
            })
    
    # Compile results
    result = {
        'top_region_indices': top_indices,
        'top_activation_values': top_activations,
        'region_labels': region_labels,
        'centralized_groups': label_groups,
        'center_regions': center_regions,
        'sorted_regions': np.array(sorted_regions),
        'sorted_activations': np.array(sorted_activations),
        'sorted_labels': np.array(sorted_labels),
        'sorted_info': sorted_info
    }
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"SUMMARY:")
    print(f"  Total regions extracted: {len(top_indices)}")
    print(f"  Number of unique labels: {len(label_groups)}")
    print(f"\nLabel Groups:")
    for label in sorted(label_groups.keys()):
        group_size = len(label_groups[label]['indices'])
        center_idx = center_regions[label]['region_index']
        center_act = center_regions[label]['activation']
        print(f"  Label '{label}': {group_size} regions, center at region {center_idx} (activation: {center_act:.6f})")
    print("="*60)
    
    return result


def calculate_region_distances(
    region_indices: np.ndarray,
    coordinate_file: str = None,
    coordinate_key: str = 'vertices'
) -> np.ndarray:
    """
    Calculate pairwise distances between regions (if coordinates are available).
    
    Parameters:
    -----------
    region_indices : np.ndarray
        Indices of regions to calculate distances for
    coordinate_file : str, optional
        Path to mat file containing region coordinates
    coordinate_key : str, optional (default='vertices')
        Key for coordinate data in the mat file
    
    Returns:
    --------
    distance_matrix : np.ndarray
        Pairwise distance matrix between regions
    """
    if coordinate_file is None:
        print("No coordinate file provided, skipping distance calculation")
        return None
    
    # Load coordinates
    coords = extract_field(coordinate_file, coordinate_key)
    
    # Get coordinates for specified regions
    region_coords = coords[region_indices]
    
    # Calculate pairwise distances
    n_regions = len(region_indices)
    distance_matrix = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            dist = np.linalg.norm(region_coords[i] - region_coords[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix


def save_centralized_results(
    result: Dict,
    output_path: str,
    include_all_info: bool = True
) -> None:
    """
    Save centralized region results to a file.
    
    Parameters:
    -----------
    result : Dict
        Result dictionary from centralize_regions_by_label
    output_path : str
        Path to save results (.txt, .csv, or .mat)
    include_all_info : bool, optional (default=True)
        Whether to include detailed information for all regions
    """
    import os
    
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.mat':
        # Save as MAT file
        save_dict = {
            'top_region_indices': result['top_region_indices'],
            'top_activation_values': result['top_activation_values'],
            'region_labels': result['region_labels'],
            'sorted_regions': result['sorted_regions'],
            'sorted_activations': result['sorted_activations'],
            'sorted_labels': result['sorted_labels']
        }
        sio.savemat(output_path, save_dict)
        
    elif ext == '.csv':
        # Save as CSV
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Region_Index', 'Activation', 'Label', 'Is_Center'])
            
            for rank, info in enumerate(result['sorted_info'], 1):
                writer.writerow([
                    rank,
                    info['region_index'],
                    f"{info['activation']:.6f}",
                    info['label'],
                    'CENTER' if info['is_center'] else ''
                ])
    
    else:  # Default to .txt
        with open(output_path, 'w') as f:
            f.write("CENTRALIZED BRAIN REGIONS BY LABEL\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total regions: {len(result['top_region_indices'])}\n")
            f.write(f"Unique labels: {len(result['centralized_groups'])}\n\n")
            
            f.write("LABEL GROUPS SUMMARY:\n")
            f.write("-"*80 + "\n")
            for label in sorted(result['centralized_groups'].keys()):
                group_size = len(result['centralized_groups'][label]['indices'])
                center_info = result['center_regions'][label]
                f.write(f"Label '{label}': {group_size} regions, ")
                f.write(f"center at region {center_info['region_index']} ")
                f.write(f"(activation: {center_info['activation']:.6f})\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED REGION LIST (Sorted by Label Groups, Centers First):\n")
            f.write("="*80 + "\n\n")
            
            current_label = None
            for rank, info in enumerate(result['sorted_info'], 1):
                if info['label'] != current_label:
                    current_label = info['label']
                    f.write(f"\n--- Label Group: {current_label} ---\n")
                
                marker = " [CENTER]" if info['is_center'] else ""
                f.write(f"{rank:4d}. Region {info['region_index']:4d} | ")
                f.write(f"Activation: {info['activation']:12.6f}{marker}\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mat_file = sys.argv[1]
        n_regions = int(sys.argv[2]) if len(sys.argv) > 2 else 70
        label_key = sys.argv[3] if len(sys.argv) > 3 else 'label'
        
        print(f"Processing: {mat_file}")
        print(f"Number of regions: {n_regions}")
        print(f"Label field: '{label_key}'")
        print("="*60 + "\n")
        
        # Process and centralize regions
        result = centralize_regions_by_label(
            mat_file,
            n_regions=n_regions,
            label_key=label_key
        )
        
        # Save results
        output_file = mat_file.replace('.mat', '_centralized.txt')
        save_centralized_results(result, output_file)
        
        # Also save as CSV for easy import
        csv_file = mat_file.replace('.mat', '_centralized.csv')
        save_centralized_results(result, csv_file)
        
    else:
        print("Usage: python region_centralization.py <mat_file_path> [n_regions] [label_key]")
        print("\nExamples:")
        print("  python region_centralization.py data.mat")
        print("  python region_centralization.py data.mat 70")
        print("  python region_centralization.py data.mat 70 ground_truth")
        print("\nThis script will:")
        print("  1. Extract the top N activated regions")
        print("  2. Group them by their labels")
        print("  3. Identify the center (max activated) region for each label group")
        print("  4. Sort regions with centers first in each group")
        print("  5. Save results to text and CSV files")
