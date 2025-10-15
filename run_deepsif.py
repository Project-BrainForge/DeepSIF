import os
import argparse
import numpy as np
import torch
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import network

def load_model(model_path, device='cpu'):
    """
    Load a pre-trained DeepSIF model.
    
    Args:
        model_path (str): Path to the model weights file (.pth or .pth.tar)
        device (str): Device to load the model on ('cpu' or 'cuda:0' etc.)
        
    Returns:
        model: Loaded DeepSIF model
    """
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    
    # Create model based on saved architecture
    if 'arch' in checkpoint and 'attribute_list' in checkpoint:
        arch = checkpoint['arch']
        attribute_list = checkpoint['attribute_list']
        model = network.__dict__[arch](*attribute_list).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        # If using simple saved state dict without architecture info
        # Using default model parameters for 75-channel EEG
        model = network.TemporalInverseNet(
            num_sensor=75,
            num_source=994,
            rnn_layer=3,
            spatial_model=network.MLPSpatialFilter,
            temporal_model=network.TemporalFilter,
            spatial_output="value_activation",
            temporal_output="rnn",
            spatial_activation="ELU",
            temporal_activation="ELU",
            temporal_input_size=500,
        ).to(device)
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded successfully with {model.count_parameters()} parameters")
    return model

def run_inference(model, eeg_data, device='cpu'):
    """
    Run inference with the DeepSIF model.
    
    Args:
        model: The loaded DeepSIF model
        eeg_data (numpy.ndarray or torch.Tensor): EEG data of shape (batch_size, time_points, channels)
                                                 or (batch_size, channels, time_points) which will be transposed
        device (str): Device to run inference on
        
    Returns:
        numpy.ndarray: Source estimates
    """
    if isinstance(eeg_data, np.ndarray):
        eeg_data = torch.from_numpy(eeg_data).float()
    
    # Check if data needs to be transposed
    # If the second dimension is larger than the third, it's likely in the format [batch, channels, time]
    # and needs to be transposed to [batch, time, channels]
    if eeg_data.shape[1] < eeg_data.shape[2]:
        print("Transposing input data from [batch, channels, time] to [batch, time, channels]")
        eeg_data = eeg_data.transpose(1, 2)
    
    eeg_data = eeg_data.to(device)
    print(f"Input shape to model: {eeg_data.shape}")
    
    with torch.no_grad():
        model_output = model(eeg_data)
        source_estimate = model_output["last"]
    
    return source_estimate.cpu().numpy()

def load_eeg_from_mat(file_path, variable_name=None):
    """
    Load EEG data from a MATLAB .mat file.
    
    Args:
        file_path (str): Path to the .mat file
        variable_name (str): Name of the variable containing EEG data. If None, will try to determine automatically.
        
    Returns:
        numpy.ndarray: EEG data of shape (batch_size, channels, time_points)
    """
    mat_data = loadmat(file_path)
    
    if variable_name is not None and variable_name in mat_data:
        eeg_data = mat_data[variable_name]
    else:
        # Try to find EEG data automatically
        for key in mat_data.keys():
            if key.startswith('__'):  # Skip metadata fields
                continue
            if isinstance(mat_data[key], np.ndarray) and mat_data[key].ndim >= 2:
                eeg_data = mat_data[key]
                variable_name = key
                print(f"Using variable '{key}' from mat file.")
                break
    
    # Check dimensions and reshape if necessary
    if eeg_data.ndim == 2:
        # Assume channels x time points, add batch dimension
        eeg_data = eeg_data[np.newaxis, :, :]
    elif eeg_data.ndim == 3:
        # Check if batch dimension is first
        if eeg_data.shape[0] > eeg_data.shape[1] and eeg_data.shape[0] > eeg_data.shape[2]:
            # Likely time x channels x instances format, transpose
            eeg_data = np.transpose(eeg_data, (2, 1, 0))
    
    return eeg_data

def save_results(source_estimate, output_path):
    """
    Save source estimation results to a .mat file.
    
    Args:
        source_estimate (numpy.ndarray): Source estimation results
        output_path (str): Path to save the results
    """
    savemat(output_path, {'source_estimate': source_estimate})
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run DeepSIF inference on EEG data")
    parser.add_argument("--model", default="model_weights/model_weights.pt", type=str, help="Path to model weights")
    parser.add_argument("--input", default="E:/DeepSIF/DeepSif -BrainForge Repo/DeepSIF/source/VEP/data1.mat", 
                        required=False, type=str, help="Path to input EEG data (.mat file)")
    parser.add_argument("--output", default="vep_source_estimate.mat", type=str, help="Path to save output results")
    parser.add_argument("--device", default="cpu", type=str, help="Device to run inference on (cpu or cuda:0)")
    parser.add_argument("--var_name", default=None, type=str, help="Variable name in the mat file containing EEG data")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Warning: Input file '{args.input}' not found.")
        return
    
    # Check if CUDA is available if requested
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        args.device = 'cpu'
    
    # Display mat file content before loading
    print(f"Inspecting MAT file: {args.input}")
    mat_contents = loadmat(args.input)
    print("MAT file variables:")
    for key in mat_contents.keys():
        if not key.startswith('__'):  # Skip metadata fields
            shape_info = f"shape={mat_contents[key].shape}" if hasattr(mat_contents[key], 'shape') else 'N/A'
            print(f"  - {key}: {type(mat_contents[key])} with {shape_info}")
    
    # Load model
    model = load_model(args.model, device=args.device)
    
    # Load EEG data
    eeg_data = load_eeg_from_mat(args.input, args.var_name)
    print(f"Loaded EEG data with shape: {eeg_data.shape}")
    
    # Run inference
    source_estimate = run_inference(model, eeg_data, device=args.device)
    print(f"Source estimation completed with shape: {source_estimate.shape}")
    
    # Save results
    save_results(source_estimate, args.output)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
