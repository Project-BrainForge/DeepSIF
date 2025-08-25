# DeepSIF Inference Script - Usage Guide

## Overview

`run_deepsif.py` is a command-line tool for running the pre-trained DeepSIF model on EEG data to perform source localization. The script handles data loading, preprocessing, inference, and saves the resulting source estimates to a .mat file.

## Features

- ✅ Supports both MATLAB v7 and v7.3+ formats
- ✅ Automatic EEG data preprocessing and normalization
- ✅ GPU acceleration support (CUDA)
- ✅ Flexible input handling with automatic dimension adjustment
- ✅ Clean command-line interface with detailed output

## Requirements

Before running the script, ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- NumPy
- SciPy
- h5py

## Basic Usage

### Minimal Command

```bash
python run_deepsif.py --eeg_file <path_to_eeg_data.mat> --var_name <variable_name>
```

### Full Command with All Options

```bash
python run_deepsif.py \
  --eeg_file <path_to_eeg_data.mat> \
  --var_name <variable_name> \
  --model <path_to_model_weights.pt> \
  --output <output_filename.mat> \
  --device <cpu|cuda> \
  --no-normalize
```

## Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--eeg_file` | string | Path to the input EEG data file in .mat format |
| `--var_name` | string | Variable name in the MAT file containing the EEG data |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | string | `model_weights/model_weights.pt` | Path to the pre-trained model weights file |
| `--output` | string | `source_estimate.mat` | Output filename for the source estimation results |
| `--device` | string | auto-detect | Device to use: `cpu` or `cuda` (auto-detects GPU if available) |
| `--no-normalize` | flag | False | Skip data normalization (not recommended) |

## Usage Examples

### Example 1: Basic Inference

```bash
python run_deepsif.py --eeg_file data.mat --var_name eeg_data
```

This will:
- Load EEG data from `data.mat` (variable: `eeg_data`)
- Use the default model at `model_weights/model_weights.pt`
- Save results to `source_estimate.mat`
- Auto-detect GPU/CPU

### Example 2: With Custom Model and Output

```bash
python run_deepsif.py \
  --eeg_file source/VEP/data1.mat \
  --var_name data \
  --model model_weights/model_weights.pt \
  --output my_results.mat
```

### Example 3: Using GPU

```bash
python run_deepsif.py \
  --eeg_file data.mat \
  --var_name eeg_data \
  --device cuda
```

### Example 4: Listing Available Variables

If you're not sure which variable contains your EEG data, run without `--var_name`:

```bash
python run_deepsif.py --eeg_file data.mat
```

The script will list all available variables in the file with their shapes and data types.

## Input Data Format

### Expected EEG Data Shape

The model expects EEG data in one of the following formats:
- `[batch, timepoints, channels]` (preferred)
- `[batch, channels, timepoints]` (will be automatically transposed)
- `[timepoints, channels]` (batch dimension will be added)
- `[channels, timepoints]` (will be transposed and batched)

### Default Dimensions

- **Channels**: 75 (will pad or truncate if different)
- **Timepoints**: 500 (will pad or truncate if different)

The script automatically adjusts your data to match these requirements.

### Preprocessing Steps

The script performs the following preprocessing (matching the training pipeline):

1. **Dimension Detection**: Automatically detects and transposes data format
2. **Channel Adjustment**: Pads with zeros or truncates to 75 channels
3. **Timepoint Adjustment**: Pads with zeros or truncates to 500 timepoints
4. **Normalization** (unless `--no-normalize` is used):
   - Remove temporal mean (across timepoints)
   - Remove spatial mean (across channels)
   - Scale to range [-1, 1]

## Output Format

The script saves results to a .mat file containing:

- `source_estimate`: Source localization estimates (numpy array)
- `shape`: Shape of the source estimate
- `description`: Description of the output

### Output Shape

- `[batch, num_sources, timepoints]` where `num_sources` is typically 994 (based on cortical parcellation)

## Model Requirements

### Default Model

The default pre-trained model should be located at:
```
model_weights/model_weights.pt
```

### Model Architecture

The script expects a model checkpoint containing either:
1. Full checkpoint with architecture information:
   - `arch`: Architecture name (e.g., 'TemporalInverseNet')
   - `attribute_list`: Model configuration parameters
   - `state_dict`: Model weights

2. Simple state dict:
   - Uses default `TemporalInverseNet` with standard parameters

### Custom Models

You can use custom trained models by specifying the `--model` parameter:

```bash
python run_deepsif.py --eeg_file data.mat --var_name eeg --model path/to/custom_model.pt
```

## Troubleshooting

### Common Issues

**Issue: "Variable not found"**
```
Solution: Run without --var_name to see available variables, then specify the correct one.
```

**Issue: "CUDA out of memory"**
```
Solution: Use CPU instead with --device cpu
```

**Issue: "Shape mismatch"**
```
Solution: The script should automatically handle most shape issues. Check that your EEG data 
is 2D or 3D with reasonable dimensions.
```

**Issue: "Model file not found"**
```
Solution: Ensure model_weights/model_weights.pt exists, or specify correct path with --model
```

### Getting Help

```bash
python run_deepsif.py --help
```

## Performance Tips

1. **Use GPU**: For faster inference, use `--device cuda` if you have a CUDA-enabled GPU
2. **Batch Processing**: The script supports batch inputs - provide data with batch dimension
3. **Pre-normalize Data**: While the script normalizes by default, consistent preprocessing improves results

## Technical Details

### Normalization

The normalization process matches the training pipeline:
```python
# Remove temporal mean
eeg_data = eeg_data - mean(eeg_data, axis=timepoints)

# Remove spatial mean
eeg_data = eeg_data - mean(eeg_data, axis=channels)

# Scale to [-1, 1]
eeg_data = eeg_data / max(abs(eeg_data))
```

### Model Output

The model returns a dictionary with the final source estimates under the key `"last"`.

## Quick Start Workflow

1. **Prepare your EEG data** in .mat format with shape `[timepoints, channels]` or `[batch, timepoints, channels]`

2. **Run inference**:
   ```bash
   python run_deepsif.py --eeg_file your_data.mat --var_name your_variable
   ```

3. **Load results** in Python or MATLAB:
   
   **Python:**
   ```python
   from scipy.io import loadmat
   results = loadmat('source_estimate.mat')
   source_est = results['source_estimate']
   ```
   
   **MATLAB:**
   ```matlab
   load('source_estimate.mat')
   % source_estimate variable is now available
   ```

## Related Files

- `network.py`: Contains the DeepSIF model architecture
- `loaders.py`: Data loading utilities used during training
- `main.py`: Training script
- `requirements.txt`: Python dependencies
- `model_weights/`: Directory containing pre-trained model weights

## Citation

If you use this script or the DeepSIF model in your research, please cite the original paper.

## Support

For issues or questions:
1. Check that all requirements are installed
2. Verify input data format matches expected dimensions
3. Review the error messages - they provide detailed diagnostics
4. Check the main README.md for general project information

---

**Last Updated**: January 2026
