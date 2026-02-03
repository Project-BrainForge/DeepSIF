"""Functions for analyzing brain region activations"""

import numpy as np
import scipy.io as sio
from typing import Union, Tuple, List


def get_top_activated_regions(
    mat_file_path: str,
    n_regions: int = 70,
    data_key: str = None,
    aggregation_method: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the top N highest activated brain regions from a MAT file.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the .mat file containing brain activation data
    n_regions : int, optional (default=70)
        Number of top activated regions to return
    data_key : str, optional
        Key name in the mat file to load. If None, will use the first numeric array found
    aggregation_method : str, optional (default='mean')
        Method to aggregate activation across time/samples:
        - 'mean': Average activation across time
        - 'max': Maximum activation across time
        - 'std': Standard deviation across time
        - 'sum': Sum of activations across time
        - 'rms': Root mean square across time
    
    Returns:
    --------
    top_region_indices : np.ndarray
        Indices of the top N activated regions (0-indexed)
    top_activation_values : np.ndarray
        Corresponding activation values for the top N regions
    
    Example:
    --------
    >>> indices, values = get_top_activated_regions('data.mat', n_regions=70)
    >>> print(f"Top region index: {indices[0]}, activation: {values[0]}")
    """
    # Load the mat file
    mat_data = sio.loadmat(mat_file_path)
    
    # Find the data array
    if data_key is None:
        # Find the first key that contains numeric data (skip metadata keys)
        for key in mat_data.keys():
            if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                data = mat_data[key]
                print(f"Using data from key: '{key}' with shape: {data.shape}")
                break
    else:
        data = mat_data[data_key]
    
    # Handle different data shapes
    # Expected: (1, 500, 994) or (500, 994) or (994,)
    data = np.squeeze(data)  # Remove singleton dimensions
    
    if data.ndim == 2:
        # Shape: (time_points, regions) or (samples, regions)
        # Aggregate across the first dimension
        if aggregation_method == 'mean':
            activation_per_region = np.mean(data, axis=0)
        elif aggregation_method == 'max':
            activation_per_region = np.max(data, axis=0)
        elif aggregation_method == 'std':
            activation_per_region = np.std(data, axis=0)
        elif aggregation_method == 'sum':
            activation_per_region = np.sum(data, axis=0)
        elif aggregation_method == 'rms':
            activation_per_region = np.sqrt(np.mean(data**2, axis=0))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    elif data.ndim == 1:
        # Shape: (regions,) - already aggregated
        activation_per_region = data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}. Expected 1D or 2D array after squeezing.")
    
    # Ensure we have the right number of regions (994)
    n_total_regions = len(activation_per_region)
    print(f"Total regions: {n_total_regions}")
    
    # Find indices of top N activated regions
    # Use absolute values for activation (to capture both positive and negative activations)
    abs_activation = np.abs(activation_per_region)
    
    # Get indices sorted by activation (descending)
    sorted_indices = np.argsort(abs_activation)[::-1]
    
    # Get top N regions
    top_region_indices = sorted_indices[:n_regions]
    top_activation_values = activation_per_region[top_region_indices]
    
    return top_region_indices, top_activation_values


def get_top_activated_regions_from_data(
    data: np.ndarray,
    n_regions: int = 70,
    aggregation_method: str = 'mean',
    top_k_values: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the top N highest activated brain regions from a numpy array.
    
    Parameters:
    -----------
    data : np.ndarray
        Brain activation data array (shape: (time_points, regions) or (regions,))
    n_regions : int, optional (default=70)
        Number of top activated regions to return
    aggregation_method : str, optional (default='mean')
        Method to aggregate activation across time/samples (if 2D data):
        - 'mean': Average activation
        - 'max': Maximum activation
        - 'std': Standard deviation
        - 'sum': Sum of activations
        - 'rms': Root mean square
        - 'top_k': Sum of top K values in each column (requires top_k_values parameter)
    top_k_values : int, optional
        When aggregation_method='top_k', this specifies how many top values
        to consider in each column (e.g., 15 means sum the highest 15 values per column)
    
    Returns:
    --------
    top_region_indices : np.ndarray
        Indices of the top N activated regions (0-indexed)
    top_activation_values : np.ndarray
        Corresponding activation values for the top N regions
    """
    data = np.squeeze(data)
    
    if data.ndim == 2:
        if aggregation_method == 'mean':
            activation_per_region = np.mean(data, axis=0)
        elif aggregation_method == 'max':
            activation_per_region = np.max(data, axis=0)
        elif aggregation_method == 'std':
            activation_per_region = np.std(data, axis=0)
        elif aggregation_method == 'sum':
            activation_per_region = np.sum(data, axis=0)
        elif aggregation_method == 'rms':
            activation_per_region = np.sqrt(np.mean(data**2, axis=0))
        elif aggregation_method == 'top_k':
            if top_k_values is None:
                raise ValueError("top_k_values must be specified when using 'top_k' aggregation method")
            # For each column, get the sum of top K values
            activation_per_region = np.zeros(data.shape[1])
            for i in range(data.shape[1]):
                col_data = np.abs(data[:, i])
                top_k_in_col = np.partition(col_data, -top_k_values)[-top_k_values:]
                activation_per_region[i] = np.sum(top_k_in_col)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    elif data.ndim == 1:
        activation_per_region = data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")
    
    abs_activation = np.abs(activation_per_region)
    sorted_indices = np.argsort(abs_activation)[::-1]
    top_region_indices = sorted_indices[:n_regions]
    top_activation_values = activation_per_region[top_region_indices]
    
    return top_region_indices, top_activation_values


def save_top_regions(
    region_indices: np.ndarray,
    activation_values: np.ndarray,
    output_path: str,
    region_names: List[str] = None
) -> None:
    """
    Save top activated regions to a file.
    
    Parameters:
    -----------
    region_indices : np.ndarray
        Indices of the top activated regions
    activation_values : np.ndarray
        Activation values for each region
    output_path : str
        Path to save the results (supports .txt, .csv, .mat)
    region_names : List[str], optional
        Names of regions corresponding to indices
    """
    import os
    
    ext = os.path.splitext(output_path)[1].lower()
    
    if ext == '.mat':
        # Save as MAT file
        save_dict = {
            'region_indices': region_indices,
            'activation_values': activation_values
        }
        if region_names:
            save_dict['region_names'] = region_names
        sio.savemat(output_path, save_dict)
        
    elif ext == '.csv':
        # Save as CSV
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            if region_names:
                writer.writerow(['Region_Index', 'Region_Name', 'Activation_Value'])
                for idx, val in zip(region_indices, activation_values):
                    writer.writerow([idx, region_names[idx], val])
            else:
                writer.writerow(['Region_Index', 'Activation_Value'])
                for idx, val in zip(region_indices, activation_values):
                    writer.writerow([idx, val])
                    
    else:  # Default to .txt
        with open(output_path, 'w') as f:
            f.write("Top Activated Brain Regions\n")
            f.write("=" * 50 + "\n\n")
            if region_names:
                f.write(f"{'Rank':<6} {'Index':<8} {'Region Name':<30} {'Activation':<12}\n")
                f.write("-" * 56 + "\n")
                for rank, (idx, val) in enumerate(zip(region_indices, activation_values), 1):
                    name = region_names[idx] if idx < len(region_names) else "Unknown"
                    f.write(f"{rank:<6} {idx:<8} {name:<30} {val:<12.6f}\n")
            else:
                f.write(f"{'Rank':<6} {'Index':<8} {'Activation':<12}\n")
                f.write("-" * 26 + "\n")
                for rank, (idx, val) in enumerate(zip(region_indices, activation_values), 1):
                    f.write(f"{rank:<6} {idx:<8} {val:<12.6f}\n")
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        mat_file = sys.argv[1]
        n_regions = int(sys.argv[2]) if len(sys.argv) > 2 else 70
        
        print(f"Processing: {mat_file}")
        print(f"Finding top {n_regions} regions...")
        
        indices, values = get_top_activated_regions(mat_file, n_regions=n_regions)
        
        print(f"\nTop {n_regions} activated regions:")
        for i, (idx, val) in enumerate(zip(indices[:10], values[:10]), 1):
            print(f"{i}. Region {idx}: {val:.6f}")
        
        if len(indices) > 10:
            print(f"... and {len(indices) - 10} more regions")
        
        # Save results
        output_file = mat_file.replace('.mat', '_top_regions.txt')
        save_top_regions(indices, values, output_file)
    else:
        print("Usage: python region_analysis.py <mat_file_path> [n_regions]")
        print("Example: python region_analysis.py data.mat 70")
