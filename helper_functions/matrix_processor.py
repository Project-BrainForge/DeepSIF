"""Functions for processing activation matrix with external labels and saving matrix results only"""

import numpy as np
import scipy.io as sio
from typing import Tuple, Optional
import sys
import os

# Handle both relative and absolute imports
try:
    from .region_analysis import get_top_activated_regions_from_data
    from .label_extraction import extract_labels, extract_field
except ImportError:
    # Add parent directory to path for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from region_analysis import get_top_activated_regions_from_data
    from label_extraction import extract_labels, extract_field


def process_activation_with_labels(
    activation_file: str,
    label_file: str,
    n_regions: int = 70,
    activation_key: str = None,
    label_key: str = 'label',
    aggregation_method: str = 'mean',
    top_k_values: int = None,
    output_file: str = None
) -> np.ndarray:
    """
    Swap columns between highly activated regions and label regions in activation matrix.
    
    Process:
    1. Find top N highly activated regions from 1x500x994 matrix
    2. Get label indices from label file (filter out any > 994)
    3. Swap columns: data at label indices <-> data at activated region indices
    4. Return modified 1x500x994 matrix
    
    Parameters:
    -----------
    activation_file : str
        Path to mat file containing 1x500x994 activation matrix
    label_file : str
        Path to mat file containing label region indices
    n_regions : int, optional (default=70)
        Number of top activated regions to find
    activation_key : str, optional
        Key for activation data in mat file (auto-detected if None)
    label_key : str, optional (default='label')
        Key for label indices in mat file
    aggregation_method : str, optional (default='mean')
        Method to aggregate activation: 'mean', 'max', 'std', 'sum', 'rms', 'top_k'
    top_k_values : int, optional
        When aggregation_method='top_k', specifies number of highest values per column to consider
    output_file : str, optional
        Path to save output mat file. If None, doesn't save to disk.
    
    Returns:
    --------
    result_matrix : np.ndarray
        Modified activation matrix with swapped columns (same shape as input)
    
    Example:
    --------
    >>> result = process_activation_with_labels('activation.mat', 'labels.mat', n_regions=70)
    >>> print(f"Result shape: {result.shape}")
    """
    print(f"Loading activation data from: {activation_file}")
    # Load activation data
    act_data = sio.loadmat(activation_file)
    
    # Find activation matrix
    if activation_key is None:
        for key in act_data.keys():
            if not key.startswith('__') and isinstance(act_data[key], np.ndarray):
                activation_matrix = act_data[key]
                print(f"Using activation key: '{key}' with shape: {activation_matrix.shape}")
                break
    else:
        activation_matrix = act_data[activation_key]
        print(f"Loaded activation shape: {activation_matrix.shape}")
    
    # Store original shape
    original_shape = activation_matrix.shape
    
    # Squeeze to work with the data
    activation_matrix = np.squeeze(activation_matrix)
    print(f"Working shape: {activation_matrix.shape}")
    
    if activation_matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix after squeezing, got shape: {activation_matrix.shape}")
    
    time_points, total_regions = activation_matrix.shape
    
    print(f"\nLoading label indices from: {label_file}")
    # Load labels (these are region indices, not labels)
    label_indices = extract_labels(label_file, label_key)
    label_indices = label_indices.astype(int).flatten()
    print(f"Label indices shape: {label_indices.shape}")
    print(f"Label indices range: [{label_indices.min()}, {label_indices.max()}]")
    
    # Filter out invalid labels (> 994 or >= total_regions)
    valid_mask = (label_indices >= 0) & (label_indices < total_regions)
    label_indices = label_indices[valid_mask]
    print(f"Valid label indices after filtering: {len(label_indices)}")
    
    if len(label_indices) == 0:
        raise ValueError("No valid label indices found after filtering!")
    
    # Get top activated regions
    print(f"\nFinding top {n_regions} activated regions...")
    if aggregation_method == 'top_k' and top_k_values:
        print(f"Using top_k method: considering highest {top_k_values} values per column")
    top_indices, top_activations = get_top_activated_regions_from_data(
        activation_matrix,
        n_regions=n_regions,
        aggregation_method=aggregation_method,
        top_k_values=top_k_values
    )
    
    print(f"Top activated regions: {top_indices[:10]}..." if len(top_indices) > 10 else f"Top activated regions: {top_indices}")
    print(f"Label indices: {label_indices[:10]}..." if len(label_indices) > 10 else f"Label indices: {label_indices}")
    
    # Determine how many swaps to make (minimum of the two arrays)
    n_swaps = min(len(top_indices), len(label_indices))
    print(f"\nPerforming {n_swaps} column swaps...")
    
    # Create a copy of the matrix to modify
    result_matrix = activation_matrix.copy()
    
    # Swap columns
    for i in range(n_swaps):
        activated_idx = top_indices[i]
        label_idx = label_indices[i]
        
        # Swap columns
        temp = result_matrix[:, activated_idx].copy()
        result_matrix[:, activated_idx] = result_matrix[:, label_idx]
        result_matrix[:, label_idx] = temp
        
        if i < 5:  # Show first few swaps
            print(f"  Swap {i+1}: Column {label_idx} <-> Column {activated_idx}")
    
    if n_swaps > 5:
        print(f"  ... and {n_swaps - 5} more swaps")
    
    # Reshape back to original shape if needed
    if len(original_shape) == 3:
        result_matrix = result_matrix.reshape(original_shape)
    
    print(f"\nResult matrix shape: {result_matrix.shape}")
    
    # Save if output file specified
    if output_file:
        print(f"\nSaving result to: {output_file}")
        save_dict = {
            'source_estimate': result_matrix,
            'swapped_pairs': np.column_stack([label_indices[:n_swaps], top_indices[:n_swaps]]),
            'label_indices_used': label_indices[:n_swaps],
            'activated_indices_used': top_indices[:n_swaps]
        }
        sio.savemat(output_file, save_dict)
        print("Saved successfully!")
    
    return result_matrix


def process_activation_single_file(
    mat_file: str,
    n_regions: int = 70,
    activation_key: str = None,
    label_key: str = 'label',
    aggregation_method: str = 'mean',
    output_file: str = None
) -> np.ndarray:
    """
    Process activation matrix and labels from the same file and return/save only the resultant matrix.
    
    Parameters:
    -----------
    mat_file : str
        Path to mat file containing both activation matrix and labels
    n_regions : int, optional (default=70)
        Number of top activated regions to extract
    activation_key : str, optional
        Key for activation data (auto-detected if None)
    label_key : str, optional (default='label')
        Key for label data
    aggregation_method : str, optional (default='mean')
        Aggregation method
    output_file : str, optional
        Path to save output mat file
    
    Returns:
    --------
    result_matrix : np.ndarray
        Processed matrix with centralized regions
    """
    print(f"Loading data from: {mat_file}")
    mat_data = sio.loadmat(mat_file)
    
    # Find activation matrix
    activation_matrix = None
    if activation_key is None:
        for key in mat_data.keys():
            if not key.startswith('__') and key != label_key and isinstance(mat_data[key], np.ndarray):
                if mat_data[key].size > 1000:  # Likely the activation matrix
                    activation_matrix = mat_data[key]
                    print(f"Using activation key: '{key}' with shape: {activation_matrix.shape}")
                    break
    else:
        activation_matrix = mat_data[activation_key]
    
    if activation_matrix is None:
        raise ValueError("Could not find activation matrix in file")
    
    # Extract labels
    labels = extract_labels(mat_file, label_key)
    
    # Get top activated regions
    print(f"\nExtracting top {n_regions} regions...")
    top_indices, top_activations = get_top_activated_regions_from_data(
        activation_matrix,
        n_regions=n_regions,
        aggregation_method=aggregation_method
    )
    
    # Get labels for top regions
    region_labels = labels[top_indices]
    
    # Group by labels
    print("Grouping regions by labels...")
    label_groups = {}
    
    for idx, activation, label in zip(top_indices, top_activations, region_labels):
        label_str = str(label)
        if label_str not in label_groups:
            label_groups[label_str] = []
        label_groups[label_str].append({
            'index': idx,
            'activation': activation
        })
    
    # Sort regions with centers first
    sorted_indices = []
    
    for label in sorted(label_groups.keys()):
        group = label_groups[label]
        sorted_group = sorted(group, key=lambda x: abs(x['activation']), reverse=True)
        sorted_indices.extend([item['index'] for item in sorted_group])
    
    sorted_indices = np.array(sorted_indices)
    
    # Extract data for these regions
    activation_matrix = np.squeeze(activation_matrix)
    
    if activation_matrix.ndim == 2:
        result_matrix = activation_matrix[:, sorted_indices]
    elif activation_matrix.ndim == 1:
        result_matrix = activation_matrix[sorted_indices]
    else:
        raise ValueError(f"Unexpected shape: {activation_matrix.shape}")
    
    print(f"\nResult matrix shape: {result_matrix.shape}")
    print(f"Number of unique labels: {len(label_groups)}")
    
    # Save if specified
    if output_file:
        print(f"\nSaving result to: {output_file}")
        sio.savemat(output_file, {
            'result_matrix': result_matrix,
            'region_indices': sorted_indices,
            'region_labels': labels[sorted_indices]
        })
        print("Saved successfully!")
    
    return result_matrix


if __name__ == "__main__":
    import sys
    import os
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process activation matrix with labels and save resultant matrix only.'
    )
    parser.add_argument('--pred', type=str, required=True,
                        help='Path to prediction/activation mat file (1x500x994)')
    parser.add_argument('--sample', type=str, required=True,
                        help='Path to sample mat file containing labels')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output mat file')
    parser.add_argument('--n_regions', type=int, default=70,
                        help='Number of top activated regions to extract (default: 70)')
    parser.add_argument('--activation_key', type=str, default=None,
                        help='Key for activation data in mat file (auto-detected if not specified)')
    parser.add_argument('--label_key', type=str, default='label',
                        help='Key for label data in mat file (default: label)')
    parser.add_argument('--aggregation', type=str, default='mean',
                        choices=['mean', 'max', 'std', 'sum', 'rms', 'top_k'],
                        help='Aggregation method for time series (default: mean)')
    parser.add_argument('--top_k_values', type=int, default=15,
                        help='Number of highest values per column to consider when using top_k aggregation (default: 15)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MATRIX PROCESSOR")
    print("="*60)
    print(f"Prediction file: {args.pred}")
    print(f"Sample file: {args.sample}")
    print(f"Output file: {args.output}")
    print(f"Number of regions: {args.n_regions}")
    print(f"Aggregation method: {args.aggregation}")
    print("="*60 + "\n")
    
    result = process_activation_with_labels(
        activation_file=args.pred,
        label_file=args.sample,
        n_regions=args.n_regions,
        activation_key=args.activation_key,
        label_key=args.label_key,
        aggregation_method=args.aggregation,
        top_k_values=args.top_k_values if args.aggregation == 'top_k' else None,
        output_file=args.output
    )
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Result matrix shape: {result.shape}")
    print(f"Saved to: {args.output}")
