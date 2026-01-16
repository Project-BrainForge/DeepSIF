# Training DeepSIF with Your Custom Labeled Dataset

## Your Dataset Format
Your .mat files contain:
- `eeg_data`: (500, 75) - EEG sensor recordings
- `source_data`: (500, 994) - Source space ground truth
- `labels`: (1, 70) - Active source region indices
- `snr`: scalar - Signal-to-noise ratio

## File Organization

Organize your dataset files like this:

```
DeepSIF/
    source/
        MyCustomData/
            train/
                data0.mat
                data1.mat
                data2.mat
                ...
            test/
                data0.mat
                data1.mat
                ...
    anatomy/
        leadfield_75_20k.mat  (your forward matrix)
```

## Step-by-Step Training Setup

### 1. Prepare Your Forward Matrix
You need a forward/leadfield matrix file (`.mat`) with key `'fwd'`:
- Shape: (num_sensors, num_sources) = (75, 994)
- Save it in the `anatomy/` folder

Example to create one:
```python
from scipy.io import savemat
import numpy as np

# If you have your forward matrix as 'fwd_matrix'
fwd_matrix = np.random.randn(75, 994)  # Replace with your actual matrix
savemat('anatomy/my_forward_matrix.mat', {'fwd': fwd_matrix})
```

### 2. Training Command

```bash
python main.py \
    --model_id 1 \
    --dat CustomLabeledDataset \
    --train MyCustomData/train \
    --test MyCustomData/test \
    --fwd my_forward_matrix.mat \
    --batch_size 32 \
    --epoch 50 \
    --lr 3e-4 \
    --workers 0
```

### 3. Key Parameters Explained

- `--dat CustomLabeledDataset`: Uses your custom data loader
- `--train`: Path relative to `source/Simulation/`
- `--test`: Path relative to `source/Simulation/`
- `--fwd`: Forward matrix filename in `anatomy/` folder
- `--model_id`: Unique ID for this training run
- `--batch_size`: Number of samples per batch (adjust based on GPU memory)
- `--epoch`: Number of training epochs
- `--lr`: Learning rate

### 4. Advanced: Custom File Naming

If your files are named differently (e.g., `sample0.mat`, `sample1.mat`), modify line 64-65 in `main.py`:

```python
train_data = loaders.__dict__[args.dat](
    data_root + args.train, 
    fwd=fwd,
    args_params={'dataset_len': 100, 'file_prefix': 'sample'}  # Customize here
)
```

### 5. Verify Data Loading

Test your data loader before training:

```python
from scipy.io import loadmat
import loaders
import numpy as np

# Load forward matrix
fwd = loadmat('anatomy/my_forward_matrix.mat')['fwd']

# Create dataset
dataset = loaders.CustomLabeledDataset(
    'source/MyCustomData/train',
    fwd=fwd,
    args_params={'dataset_len': 10}
)

# Test loading first sample
sample = dataset[0]
print(f"Data shape: {sample['data'].shape}")        # Should be (500, 75)
print(f"NMM shape: {sample['nmm'].shape}")          # Should be (500, 994)
print(f"Labels: {sample['label']}")                 # Your active regions
print(f"SNR: {sample['snr']}")                      # SNR value
```

## Expected Output Structure

Training will create:
```
model_result/
    1_the_model/          (your model_id)
        model_best.pth.tar
        epoch_1
        epoch_2
        ...
        outputs_TemporalInverseNet.log
        train_test_error.mat
```

## Troubleshooting

### Issue: Dataset length is 0
- Check file paths are correct
- Ensure files are named `data0.mat`, `data1.mat`, etc.
- Or specify `dataset_len` manually in args_params

### Issue: Shape mismatch
- Verify `eeg_data` is (500, 75)
- Verify `source_data` is (500, 994)
- Check forward matrix is (75, 994)

### Issue: Out of memory
- Reduce `--batch_size` (try 16 or 8)
- Use `--workers 0` to disable multiprocessing

### Issue: Label padding
If your labels contain padding values, modify line 91 in the custom loader:
```python
# Remove padding (adjust the condition based on your padding value)
labels = labels[labels != padding_value]
```

## Next Steps

After training:
1. Monitor `train_test_error.mat` for convergence
2. Use `eval_sim.py` to evaluate on test data
3. Visualize results with `misc_scripts/visualize_result.m`
