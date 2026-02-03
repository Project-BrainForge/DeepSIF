"""
DeepSIF Inference Script
Unified script for running pre-trained DeepSIF model on EEG data for source localization.
This combines and improves upon example_inference.py and the original run_deepsif.py.
"""

import os
import argparse
import numpy as np
import torch
from scipy.io import loadmat, savemat
import h5py
import network


def load_mat_file(filepath, variable_name=None):
    """
    Load .mat file supporting both MATLAB v7 and v7.3+ formats.
    
    Args:
        filepath (str): Path to .mat file
        variable_name (str): Optional variable name to extract
        
    Returns:
        numpy.ndarray or dict: EEG data or dictionary with file contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If variable_name is specified but not found
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        # Try scipy first (MATLAB v7 and earlier)
        data = loadmat(filepath)
        print(f"✓ Loaded using scipy.io (MATLAB v7 format)")
    except (ValueError, NotImplementedError, OSError):
        # Fall back to h5py (MATLAB v7.3+)
        print(f"✓ Loading using h5py (MATLAB v7.3+ format)")
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                if not key.startswith(('#', '__')):
                    try:
                        data[key] = np.array(f[key])
                    except:
                        print(f"  Warning: Could not load variable '{key}'")
        print("data",data)
    
    # Display available variables
    print("\nAvailable variables in MAT file:")
    for key, val in data.items():
        if not key.startswith(('#', '__')):
            shape = val.shape if hasattr(val, 'shape') else 'N/A'
            dtype = val.dtype if hasattr(val, 'dtype') else type(val).__name__
            print(f"  - {key}: shape {shape}, dtype {dtype}")
    
    # Extract specific variable if requested
    if variable_name:
        if variable_name not in data:
            available = [k for k in data.keys() if not k.startswith(('#', '__'))]
            raise ValueError(f"Variable '{variable_name}' not found. Available: {available}")
        return data[variable_name]
    
    print("finally return data ----- ",data)
    
    return data


def preprocess_eeg(eeg_data, target_channels=75, target_timepoints=500, normalize=True):
    """
    Preprocess EEG data to match model requirements.
    Matches the preprocessing in loaders.py (lines 86-88).
    
    Args:
        eeg_data (numpy.ndarray): Input EEG array
        target_channels (int): Expected number of channels (default: 75)
        target_timepoints (int): Expected number of timepoints (default: 500)
        normalize (bool): Whether to normalize data (default: True)
        
    Returns:
        numpy.ndarray: Preprocessed EEG data with shape [batch, timepoints, channels]
    """
    print(f"\n📊 Preprocessing EEG data...")
    print(f"  Input shape: {eeg_data.shape}")
    
    # Add batch dimension if missing
    if eeg_data.ndim == 2:
        eeg_data = eeg_data[np.newaxis, ...]
        print(f"  Added batch dimension")
    
    # Detect and transpose if needed
    # Check if data is in [batch, channels, time] format
    if eeg_data.shape[1] == target_channels or (eeg_data.shape[1] < eeg_data.shape[2]):
        # Likely [batch, channels, time] -> transpose to [batch, time, channels]
        eeg_data = np.transpose(eeg_data, (0, 2, 1))
        print(f"  Transposed from [batch, channels, time] to [batch, time, channels]")
    
    # Now should be [batch, time, channels]
    batch, time, channels = eeg_data.shape
    
    # Adjust channels
    if channels != target_channels:
        print(f"  Adjusting channels: {channels} → {target_channels}")
        if channels < target_channels:
            pad_width = ((0, 0), (0, 0), (0, target_channels - channels))
            eeg_data = np.pad(eeg_data, pad_width, mode='constant', constant_values=0)
        else:
            eeg_data = eeg_data[:, :, :target_channels]
    
    # Adjust timepoints
    if time != target_timepoints:
        print(f"  Adjusting timepoints: {time} → {target_timepoints}")
        if time < target_timepoints:
            pad_width = ((0, 0), (0, target_timepoints - time), (0, 0))
            eeg_data = np.pad(eeg_data, pad_width, mode='constant', constant_values=0)
        else:
            eeg_data = eeg_data[:, :target_timepoints, :]
    
    # Normalize (matches loaders.py lines 86-88)
    if normalize:
        print(f"  Normalizing data...")
        eeg_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)  # Remove time mean
        eeg_data = eeg_data - np.mean(eeg_data, axis=2, keepdims=True)  # Remove channel mean
        max_val = np.max(np.abs(eeg_data))
        if max_val > 0:
            eeg_data = eeg_data / max_val  # Scale to [-1, 1]
    
    print(f"  ✓ Preprocessed shape: {eeg_data.shape}")
    return eeg_data


def load_model(model_path, device='cpu'):
    """
    Load a pre-trained DeepSIF model.
    
    Args:
        model_path (str): Path to the model weights file (.pt, .pth, or .pth.tar)
        device (str): Device to load the model on ('cpu' or 'cuda:0' etc.)
        
    Returns:
        model: Loaded DeepSIF model in evaluation mode
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"\n🧠 Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    
    # Create model based on saved architecture
    if 'arch' in checkpoint and 'attribute_list' in checkpoint:
        arch = checkpoint['arch']
        attribute_list = checkpoint['attribute_list']
        model = network.__dict__[arch](*attribute_list).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"  ✓ Loaded {arch} architecture from checkpoint")
    else:
        # If using simple saved state dict without architecture info
        # Using default model parameters for 75-channel EEG
        print(f"  Using default TemporalInverseNet architecture")
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
    print(f"  ✓ Model loaded successfully with {model.count_parameters()} parameters")
    return model


def run_inference(model, eeg_data, device='cpu'):
    """
    Run inference with the DeepSIF model.
    
    Args:
        model: The loaded DeepSIF model
        eeg_data (numpy.ndarray): Preprocessed EEG data of shape [batch, time, channels]
        device (str): Device to run inference on
        
    Returns:
        numpy.ndarray: Source estimates
    """
    print(f"\n🔬 Running inference...")
    
    # Convert to tensor
    if isinstance(eeg_data, np.ndarray):
        eeg_data = torch.from_numpy(eeg_data).float()
    
    eeg_data = eeg_data.to(device)
    print(f"  Input shape: {eeg_data.shape}")
    
    # Run inference
    with torch.no_grad():
        model_output = model(eeg_data)
        source_estimate = model_output["last"]
    
    result = source_estimate.cpu().numpy()
    print(f"  ✓ Output shape: {result.shape}")
    
    return result

def save_results(source_estimate, output_path):
    """
    Save source estimation results to a .mat file.
    
    Args:
        source_estimate (numpy.ndarray): Source estimation results
        output_path (str): Path to save the results
    """
    savemat(output_path, {
        'source_estimate': source_estimate,
        'shape': source_estimate.shape,
        'description': 'DeepSIF source localization results'
    })
    print(f"💾 Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="DeepSIF: Run pre-trained model on EEG data for source localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with variable name specified
  python run_deepsif.py --eeg_file data.mat --var_name eeg_data
  
  # With custom model and GPU
  python run_deepsif.py --eeg_file data.mat --var_name eeg_data --model custom_weights.pt --device cuda
  
  # Full example
  python run_deepsif.py --eeg_file source/VEP/data1.mat --var_name data --model model_weights/model_weights.pt --output results.mat
        """
    )
    
    parser.add_argument("--eeg_file", type=str, required=True,
                       help="Path to EEG data file (.mat)")
    parser.add_argument("--var_name", type=str, default=None,
                       help="Variable name in MAT file containing EEG data (optional, will list available if not provided)")
    parser.add_argument("--model", type=str, default="model_weights/model_weights.pt",
                       help="Path to model weights (default: model_weights/model_weights.pt)")
    parser.add_argument("--output", type=str, default="source_estimate.mat",
                       help="Output filename for results (default: source_estimate.mat)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use: 'cpu' or 'cuda' (default: auto-detect)")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Skip normalization step (not recommended)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("  DeepSIF - Deep Source Imaging of Focal sources")
    print("="*60)
    
    # Set device
    if args.device:
        device = args.device
        if device.startswith('cuda') and not torch.cuda.is_available():
            print("⚠ CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n⚙️  Device: {device.upper()}")
    
    try:
        # Load EEG data
        print(f"\n📁 Loading EEG data from: {args.eeg_file}")
        eeg_data = load_mat_file(args.eeg_file, args.var_name)
        
        # Check if we need to prompt for variable name
        if args.var_name is None:
            print("\n⚠  Please specify --var_name to select which variable to use for inference")
            return
        
        # Preprocess data
        eeg_data = preprocess_eeg(eeg_data, normalize=not args.no_normalize)
        
        # Load model
        model = load_model(args.model, device=device)
        
        # Run inference
        source_estimate = run_inference(model, eeg_data, device=device)
        
        # Save results
        save_results(source_estimate, args.output)
        
        print("\n" + "="*60)
        print("  ✓ Inference completed successfully!")
        print("="*60)
        print(f"  Input:  {args.eeg_file}")
        print(f"  Output: {args.output}")
        print(f"  Shape:  {source_estimate.shape}")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
