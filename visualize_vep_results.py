"""
Visualize VEP Results from rnn_test_1_.pth.tar.mat
Data shape: 3 samples x 500 timepoints x 994 vertices
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Load data
print("Loading data...")
data = sio.loadmat('source/VEP/rnn_test_1_.pth.tar.mat')
all_out = data['all_out']  # Shape: (3, 500, 994)

print(f"Data shape: {all_out.shape}")
print(f"Value range: [{all_out.min():.4f}, {all_out.max():.4f}]")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

#%% 1. Spatiotemporal Heatmaps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for sample in range(3):
    data_sample = all_out[sample, :, :]  # 500 x 994
    
    im = axes[sample].imshow(data_sample.T, aspect='auto', cmap='RdBu_r', 
                             interpolation='bilinear')
    axes[sample].set_xlabel('Time (samples)', fontsize=12)
    axes[sample].set_ylabel('Vertex Index', fontsize=12)
    axes[sample].set_title(f'Sample {sample+1}', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[sample], label='Activation')

plt.suptitle('Spatiotemporal Activation Patterns', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('vep_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 2. Time Series of Peak Vertices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = plt.cm.viridis(np.linspace(0, 1, 5))

for sample in range(3):
    data_sample = all_out[sample, :, :]  # 500 x 994
    
    # Find top 5 most active vertices
    max_activity = np.max(np.abs(data_sample), axis=0)
    top_indices = np.argsort(max_activity)[-5:][::-1]
    
    for i, vertex_idx in enumerate(top_indices):
        axes[sample].plot(data_sample[:, vertex_idx], 
                         label=f'Vertex {vertex_idx}',
                         linewidth=2, alpha=0.8, color=colors[i])
    
    axes[sample].set_xlabel('Time (samples)', fontsize=12)
    axes[sample].set_ylabel('Activation', fontsize=12)
    axes[sample].set_title(f'Sample {sample+1} - Top 5 Vertices', 
                          fontsize=14, fontweight='bold')
    axes[sample].legend(loc='best', fontsize=9)
    axes[sample].grid(True, alpha=0.3)

plt.suptitle('Time Series of Most Active Vertices', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('vep_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 3. Statistical Summary Across Samples
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Mean activation
mean_activation = np.mean(all_out, axis=0)  # 500 x 994
im1 = axes[0, 0].imshow(mean_activation.T, aspect='auto', cmap='hot', 
                        interpolation='bilinear')
axes[0, 0].set_xlabel('Time (samples)', fontsize=12)
axes[0, 0].set_ylabel('Vertex Index', fontsize=12)
axes[0, 0].set_title('Mean Activation (across 3 samples)', fontsize=13, fontweight='bold')
plt.colorbar(im1, ax=axes[0, 0], label='Mean Activation')

# Standard deviation
std_activation = np.std(all_out, axis=0)  # 500 x 994
im2 = axes[0, 1].imshow(std_activation.T, aspect='auto', cmap='YlOrRd', 
                        interpolation='bilinear')
axes[0, 1].set_xlabel('Time (samples)', fontsize=12)
axes[0, 1].set_ylabel('Vertex Index', fontsize=12)
axes[0, 1].set_title('Std Dev (across 3 samples)', fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=axes[0, 1], label='Std Dev')

# Global time course (spatial average)
spatial_mean = np.mean(all_out, axis=2)  # 3 x 500
for sample in range(3):
    axes[1, 0].plot(spatial_mean[sample, :], 
                   label=f'Sample {sample+1}', linewidth=2, alpha=0.8)
axes[1, 0].set_xlabel('Time (samples)', fontsize=12)
axes[1, 0].set_ylabel('Mean Activation', fontsize=12)
axes[1, 0].set_title('Global Time Course (spatial average)', fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Peak activation distribution across time
peak_times = np.argmax(np.abs(all_out), axis=1)  # 3 x 994
axes[1, 1].hist([peak_times[0, :], peak_times[1, :], peak_times[2, :]], 
                bins=30, alpha=0.6, label=['Sample 1', 'Sample 2', 'Sample 3'])
axes[1, 1].set_xlabel('Time of Peak Activation (samples)', fontsize=12)
axes[1, 1].set_ylabel('Number of Vertices', fontsize=12)
axes[1, 1].set_title('Distribution of Peak Activation Times', fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vep_statistics.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 4. Compare Samples at Specific Time Points
time_points = [100, 250, 400]  # ms or samples
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for i, t in enumerate(time_points):
    for sample in range(3):
        activation_at_t = all_out[sample, t, :]  # 994 vertices
        
        axes[sample, i].bar(range(len(activation_at_t)), activation_at_t, 
                           color='steelblue', alpha=0.7, width=1.0)
        axes[sample, i].set_xlabel('Vertex Index', fontsize=10)
        axes[sample, i].set_ylabel('Activation', fontsize=10)
        axes[sample, i].set_title(f'Sample {sample+1}, t={t}', fontsize=11)
        axes[sample, i].grid(True, alpha=0.3, axis='y')

plt.suptitle('Spatial Distribution at Specific Time Points', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('vep_snapshots.png', dpi=300, bbox_inches='tight')
plt.show()

#%% 5. 3D Surface Plot (optional - for single timepoint)
# This creates a 3D visualization of activation over space and time
fig = plt.figure(figsize=(15, 5))

for sample in range(3):
    ax = fig.add_subplot(1, 3, sample+1, projection='3d')
    
    # Downsample for visualization
    time_ds = all_out[sample, ::10, :]  # Every 10th timepoint
    vertex_ds = time_ds[:, ::20]  # Every 20th vertex
    
    X, Y = np.meshgrid(range(vertex_ds.shape[1]), range(vertex_ds.shape[0]))
    
    surf = ax.plot_surface(X, Y, vertex_ds, cmap='coolwarm', 
                          linewidth=0, antialiased=True, alpha=0.8)
    
    ax.set_xlabel('Vertex Index (subsampled)', fontsize=10)
    ax.set_ylabel('Time (subsampled)', fontsize=10)
    ax.set_zlabel('Activation', fontsize=10)
    ax.set_title(f'Sample {sample+1}', fontsize=12, fontweight='bold')
    ax.view_init(elev=30, azim=45)

plt.suptitle('3D Spatiotemporal Activation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('vep_3d_surface.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ All visualizations saved!")
print("  - vep_heatmaps.png")
print("  - vep_timeseries.png")
print("  - vep_statistics.png")
print("  - vep_snapshots.png")
print("  - vep_3d_surface.png")
