"""Functions for extracting labels and fields from MAT files"""

import numpy as np
import scipy.io as sio
from typing import Union, Dict, Any, List


def extract_labels(mat_file_path: str, label_key: str = 'label') -> np.ndarray:
    """
    Extract label field from a MAT file.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the .mat file containing label data
    label_key : str, optional (default='label')
        Key name in the mat file for the label field. Common variations:
        'label', 'labels', 'Label', 'ground_truth', 'gt', etc.
    
    Returns:
    --------
    labels : np.ndarray
        Extracted label array
    
    Example:
    --------
    >>> labels = extract_labels('data.mat')
    >>> print(f"Labels shape: {labels.shape}")
    """
    # Load the mat file
    mat_data = sio.loadmat(mat_file_path)
    
    # Try to find the label field
    if label_key in mat_data:
        labels = mat_data[label_key]
        print(f"Found '{label_key}' with shape: {labels.shape}")
    else:
        # Try common variations
        possible_keys = ['label', 'labels', 'Label', 'Labels', 'LABEL', 
                        'ground_truth', 'gt', 'GT', 'y', 'target']
        
        found = False
        for key in possible_keys:
            if key in mat_data:
                labels = mat_data[key]
                print(f"Found '{key}' (using as label field) with shape: {labels.shape}")
                found = True
                break
        
        if not found:
            # List available keys
            available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            raise KeyError(f"Label field '{label_key}' not found in mat file. "
                          f"Available keys: {available_keys}")
    
    # Squeeze to remove singleton dimensions
    labels = np.squeeze(labels)
    
    return labels


def extract_field(mat_file_path: str, field_key: str) -> Union[np.ndarray, Any]:
    """
    Extract any specific field from a MAT file.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the .mat file
    field_key : str
        Key name of the field to extract
    
    Returns:
    --------
    field_data : np.ndarray or Any
        Extracted field data
    
    Example:
    --------
    >>> data = extract_field('data.mat', 'eeg_data')
    """
    mat_data = sio.loadmat(mat_file_path)
    
    if field_key not in mat_data:
        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        raise KeyError(f"Field '{field_key}' not found. Available keys: {available_keys}")
    
    field_data = mat_data[field_key]
    print(f"Extracted '{field_key}' with shape: {field_data.shape if hasattr(field_data, 'shape') else type(field_data)}")
    
    return field_data


def list_mat_fields(mat_file_path: str) -> List[str]:
    """
    List all available fields in a MAT file.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the .mat file
    
    Returns:
    --------
    field_names : List[str]
        List of field names (excluding metadata fields starting with '__')
    
    Example:
    --------
    >>> fields = list_mat_fields('data.mat')
    >>> print("Available fields:", fields)
    """
    mat_data = sio.loadmat(mat_file_path)
    field_names = [k for k in mat_data.keys() if not k.startswith('__')]
    
    print(f"\nAvailable fields in {mat_file_path}:")
    for i, field in enumerate(field_names, 1):
        data = mat_data[field]
        shape_str = str(data.shape) if hasattr(data, 'shape') else str(type(data))
        print(f"  {i}. '{field}' - {shape_str}")
    
    return field_names


def extract_multiple_fields(mat_file_path: str, field_keys: List[str]) -> Dict[str, Any]:
    """
    Extract multiple fields from a MAT file at once.
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the .mat file
    field_keys : List[str]
        List of field names to extract
    
    Returns:
    --------
    fields_dict : Dict[str, Any]
        Dictionary mapping field names to their data
    
    Example:
    --------
    >>> data = extract_multiple_fields('data.mat', ['label', 'eeg_data', 'time'])
    >>> labels = data['label']
    """
    mat_data = sio.loadmat(mat_file_path)
    fields_dict = {}
    
    for field_key in field_keys:
        if field_key in mat_data:
            fields_dict[field_key] = mat_data[field_key]
            print(f"Extracted '{field_key}' with shape: {mat_data[field_key].shape if hasattr(mat_data[field_key], 'shape') else type(mat_data[field_key])}")
        else:
            print(f"Warning: Field '{field_key}' not found, skipping.")
    
    return fields_dict


def save_labels(labels: np.ndarray, output_path: str, format: str = 'txt') -> None:
    """
    Save extracted labels to a file.
    
    Parameters:
    -----------
    labels : np.ndarray
        Label array to save
    output_path : str
        Path to save the labels
    format : str, optional (default='txt')
        Output format: 'txt', 'csv', 'npy', or 'mat'
    """
    import os
    
    # Ensure correct extension
    base_path = os.path.splitext(output_path)[0]
    
    if format == 'txt':
        output_path = base_path + '.txt'
        np.savetxt(output_path, labels, fmt='%s')
    elif format == 'csv':
        output_path = base_path + '.csv'
        np.savetxt(output_path, labels, delimiter=',', fmt='%s')
    elif format == 'npy':
        output_path = base_path + '.npy'
        np.save(output_path, labels)
    elif format == 'mat':
        output_path = base_path + '.mat'
        sio.savemat(output_path, {'labels': labels})
    else:
        raise ValueError(f"Unknown format: {format}. Use 'txt', 'csv', 'npy', or 'mat'")
    
    print(f"Labels saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mat_file = sys.argv[1]
        label_key = sys.argv[2] if len(sys.argv) > 2 else 'label'
        
        print(f"Processing: {mat_file}")
        print(f"Looking for label field: '{label_key}'")
        print("=" * 60)
        
        # First, list all available fields
        print("\nStep 1: Listing all fields in the file...")
        fields = list_mat_fields(mat_file)
        
        # Try to extract labels
        print(f"\nStep 2: Attempting to extract labels...")
        try:
            labels = extract_labels(mat_file, label_key)
            print(f"\nSuccessfully extracted labels!")
            print(f"Shape: {labels.shape}")
            print(f"Data type: {labels.dtype}")
            
            # Show first few labels
            if labels.ndim == 1:
                print(f"\nFirst 10 labels: {labels[:10]}")
                if len(labels) > 10:
                    print(f"... and {len(labels) - 10} more")
            else:
                print(f"\nLabel array preview:\n{labels}")
            
            # Save labels
            output_file = mat_file.replace('.mat', '_labels.txt')
            save_labels(labels, output_file)
            
        except KeyError as e:
            print(f"\nError: {e}")
            print("\nPlease specify the correct field name:")
            print(f"Usage: python label_extraction.py <mat_file> <label_field_name>")
    else:
        print("Usage: python label_extraction.py <mat_file_path> [label_key]")
        print("\nExamples:")
        print("  python label_extraction.py data.mat")
        print("  python label_extraction.py data.mat ground_truth")
        print("\nThis will:")
        print("  1. List all available fields in the mat file")
        print("  2. Extract the specified label field (default: 'label')")
        print("  3. Save labels to a text file")
