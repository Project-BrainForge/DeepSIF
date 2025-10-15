import torch
import numpy as np
import os
import argparse
import h5py
import scipy.io
from run_deepsif import load_model, run_inference, save_results, load_eeg_from_mat

def load_mat_file(filepath):
    """Load .mat file, handling MATLAB, HDF5, and Octave text formats"""
    # First check if it's an Octave text format file
    try:
        with open(filepath, 'rb') as f:
            header = f.read(20)
            if b'Created by Octave' in header:
                print("Detected Octave text format, converting to binary...")
                print("ERROR: This file was saved in Octave's text format.")
                print("Please re-save it in Octave using: save('-v7', 'filename.mat', 'variable_name')")
                print("Or use: save('-binary', 'filename.mat', 'variable_name')")
                raise ValueError("Octave text format not supported. Please re-save in binary format.")
    except Exception as e:
        if "text format not supported" in str(e):
            raise
    
    try:
        # Try loading with scipy (for MATLAB v7 and earlier)
        return scipy.io.loadmat(filepath)
    except (ValueError, NotImplementedError, OSError):
        # If that fails, try h5py (for MATLAB v7.3 and later)
        print("Detected MATLAB v7.3+ format, using h5py...")
        mat_dict = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                if not key.startswith('#'):  # Skip HDF5 metadata
                    try:
                        mat_dict[key] = np.array(f[key])
                    except:
                        mat_dict[key] = f[key]
        return mat_dict

def main():
    parser = argparse.ArgumentParser(description="Run DeepSIF on EEG data")
    parser.add_argument("--eeg_file", type=str, 
                       default="E:/DeepSIF/DeepSif -BrainForge Repo/DeepSIF/source/VEP/data1.mat", 
                       help="Path to EEG data file (.mat)")
    parser.add_argument("--var_name", type=str, help="Variable name in the mat file containing EEG data")
    parser.add_argument("--model", type=str, 
                       default="E:/DeepSIF/DeepSif -BrainForge Repo/DeepSIF/model_weights/model_weights.pt", 
                       help="Path to model weights")
    parser.add_argument("--output", type=str, default="vep_source_estimate.mat", help="Output filename")
    args = parser.parse_args()

    # Path to model weights
    MODEL_PATH = args.model

    # Select device (CPU or GPU)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load EEG data
    if args.eeg_file and os.path.exists(args.eeg_file):
        print(f"Loading EEG data from {args.eeg_file}")
        try:
            # First, inspect the MAT file structure to help identify the right variable
            mat_contents = load_mat_file(args.eeg_file)
            print("MAT file variables:")
            for key in mat_contents.keys():
                if not key.startswith('__') and not key.startswith('#'):  # Skip metadata fields
                    print(f"  - {key}: {type(mat_contents[key])} with shape {mat_contents[key].shape if hasattr(mat_contents[key], 'shape') else 'N/A'}")
            
            # Load the actual data
            eeg_data = load_eeg_from_mat(args.eeg_file, args.var_name)
            print(f"Loaded EEG data with shape: {eeg_data.shape}")
            
            # Check if we have the expected 75 channels
            if eeg_data.shape[1] != 75 and eeg_data.shape[2] != 75:
                print(f"Warning: Expected 75 channels but found {eeg_data.shape[1]} (or {eeg_data.shape[2]}) channels.")
                print("The model may not work correctly if the channel count doesn't match.")
            
            # Ensure the data has the correct time dimension
            # The model expects 500 time points, resize if necessary
            required_time_points = 500
            current_time_dim = eeg_data.shape[2]  # Assuming [batch, channels, time]
            
            if current_time_dim != required_time_points:
                print(f"Resizing time dimension from {current_time_dim} to {required_time_points}")
                if current_time_dim > required_time_points:
                    # Truncate
                    eeg_data = eeg_data[:, :, :required_time_points]
                else:
                    # Pad with zeros
                    pad_size = required_time_points - current_time_dim
                    eeg_data = np.pad(eeg_data, ((0, 0), (0, 0), (0, pad_size)), 'constant')
                
                print(f"After resizing: {eeg_data.shape}")
            
        except Exception as e:
            print(f"Error loading EEG data: {e}")
            print("Using random test data instead.")
            eeg_data = np.random.randn(1, 75, 500)
    else:
        print("No EEG file provided or file not found. Using random test data instead.")
        # Create random example EEG data (75 channels, 500 time points)
        eeg_data = np.random.randn(1, 75, 500)  # [batch_size, channels, time_points]
        print(f"Generated random EEG data with shape: {eeg_data.shape}")

    # Reshape the data to match model expectations
    # The model expects [batch_size, time_points, channels] instead of [batch_size, channels, time_points]
    eeg_data = np.transpose(eeg_data, (0, 2, 1))
    print(f"Input data shape after reshaping: {eeg_data.shape}")
    
    # Verify dimensions are correct for this specific model
    if eeg_data.shape[1] != 500 or eeg_data.shape[2] != 75:
        print(f"Warning: Expected shape [batch, 500, 75] but got {eeg_data.shape}")
        print("Adjusting dimensions to match model requirements...")
        
        # Ensure we have 75 channels
        if eeg_data.shape[2] != 75:
            if eeg_data.shape[2] < 75:
                # Pad with zeros if we have fewer channels
                pad_channels = 75 - eeg_data.shape[2]
                eeg_data = np.pad(eeg_data, ((0, 0), (0, 0), (0, pad_channels)), 'constant')
            else:
                # Truncate if we have more channels
                eeg_data = eeg_data[:, :, :75]
        
        # Ensure we have 500 time points
        if eeg_data.shape[1] != 500:
            if eeg_data.shape[1] < 500:
                # Pad with zeros if we have fewer time points
                pad_time = 500 - eeg_data.shape[1]
                eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_time), (0, 0)), 'constant')
            else:
                # Truncate if we have more time points
                eeg_data = eeg_data[:, :500, :]
        
        print(f"Adjusted data shape: {eeg_data.shape}")

    # Load the model
    model = load_model(MODEL_PATH, device=device)

    # Run inference
    source_estimate = run_inference(model, eeg_data, device=device)
    print(f"Source estimation shape: {source_estimate.shape}")

    # Save the results
    save_results(source_estimate, args.output)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
