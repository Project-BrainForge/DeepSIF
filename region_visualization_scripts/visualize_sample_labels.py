import pyvista as pv
import scipy.io
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_sample_file(sample_filepath):
    """
    Load and examine the structure of a sample .mat file
    
    Parameters:
    -----------
    sample_filepath : str
        Path to the sample .mat file
        
    Returns:
    --------
    dict : Contents of the .mat file
    """
    try:
        data = scipy.io.loadmat(sample_filepath)
        print(f"✓ Loaded sample file: {sample_filepath}")
        print(f"\n📋 File contents:")
        for key in data.keys():
            if not key.startswith('__'):
                value = data[key]
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {value.shape} {value.dtype}")
                else:
                    print(f"  {key}: {type(value)}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: Could not find file {sample_filepath}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def visualize_sample_labels(sample_filepath, highlight_color='red', background_color='lightgray', 
                            show_statistics=True, opacity=1.0):
    """
    Visualize the labels from a sample .mat file on the cortical surface
    
    Parameters:
    -----------
    sample_filepath : str
        Path to the sample .mat file containing labels
    highlight_color : str
        Color for the highlighted regions (default: 'red')
    background_color : str
        Color for all other regions (default: 'lightgray')
    show_statistics : bool
        Whether to show statistics overlay
    opacity : float
        Opacity of the highlighted regions (0.0 to 1.0)
    """
    
    print(f"🧠 Sample Labels Visualizer")
    print("=" * 60)
    
    # Load the sample file
    sample_data = load_sample_file(sample_filepath)
    if sample_data is None:
        return
    
    # Extract labels from the sample file
    # Based on the screenshot, it looks like the file contains 'labels' variable
    if 'labels' not in sample_data:
        print(f"\n❌ Error: 'labels' not found in the sample file")
        print(f"Available keys: {[k for k in sample_data.keys() if not k.startswith('__')]}")
        return
    
    labels = sample_data['labels'].flatten()
    print(f"\n📊 Label information:")
    print(f"  Total labels: {len(labels)}")
    print(f"  Unique labels: {np.unique(labels)}")
    print(f"  Label range: {labels.min()} to {labels.max()}")
    
    # Load cortex and mapping data
    print(f"\n🔧 Loading cortical surface and region mapping...")
    try:
        cortex = scipy.io.loadmat("../anatomy/fs_cortex_20k.mat")
        region_map = scipy.io.loadmat("../anatomy/fs_cortex_20k_region_mapping.mat")
    except FileNotFoundError:
        print("❌ Error: Could not find the required .mat files in ../anatomy/")
        print("   Make sure the anatomy files are in the correct location")
        return
    
    pos = cortex['pos']  # (20484, 3) - vertex positions
    tri = cortex['tri'] - 1  # Convert to 0-based indexing
    rm = region_map['rm'].flatten()  # (20484,) - region ID per vertex
    
    print(f"✓ Loaded {pos.shape[0]} vertices and {tri.shape[0]} triangles")
    
    # Create a binary mask for visualization
    # Vertices that belong to any of the labeled regions
    color_mask = np.zeros(len(rm), dtype=int)
    
    region_stats = []
    total_vertices_highlighted = 0
    
    print(f"\n🎯 Analyzing labeled regions:")
    for label in labels:
        label_int = int(label)
        target_mask = rm == label_int
        vertex_count = np.sum(target_mask)
        
        if vertex_count > 0:
            # Mark these vertices for highlighting (1 = highlighted)
            color_mask[target_mask] = 1
            total_vertices_highlighted += vertex_count
            
            # Calculate statistics
            region_vertices = pos[target_mask]
            center = np.mean(region_vertices, axis=0)
            extent = np.ptp(region_vertices, axis=0)
            
            region_stats.append({
                'label': label_int,
                'vertices': vertex_count,
                'center': center,
                'extent': extent,
                'percentage': 100 * vertex_count / len(rm)
            })
            
            print(f"  ✓ Label {label_int}: {vertex_count:,} vertices ({100 * vertex_count / len(rm):.2f}%)")
        else:
            print(f"  ⚠ Label {label_int}: Not found in region mapping")
    
    # Create mesh
    mesh = pv.PolyData(pos, np.hstack([np.full((tri.shape[0], 1), 3), tri]).astype(np.int32))
    mesh["Label_Mask"] = color_mask
    mesh["Region_ID"] = rm
    
    # Create visualization
    print(f"\n🖥️  Creating visualization...")
    plotter = pv.Plotter(window_size=(1400, 1000))
    
    # Add mesh with custom coloring
    # Use a simple 2-color colormap: background and highlighted
    cmap_colors = [background_color, highlight_color]
    
    plotter.add_mesh(
        mesh,
        scalars="Label_Mask",
        cmap=cmap_colors,
        show_scalar_bar=False,
        opacity=opacity
    )
    
    # Create title
    sample_name = sample_filepath.split('/')[-1].split('\\')[-1]
    title = f"Sample File: {sample_name}\n"
    title += f"{len(labels)} labeled regions, {total_vertices_highlighted:,} vertices "
    title += f"({100 * total_vertices_highlighted / len(rm):.2f}% of surface)"
    plotter.add_title(title, font_size=14)
    
    # Set camera for good view
    plotter.camera_position = 'iso'
    
    # Add region center markers
    if show_statistics and region_stats:
        for stat in region_stats:
            plotter.add_mesh(
                pv.Sphere(radius=1.5, center=stat['center']), 
                color=highlight_color, 
                opacity=0.8
            )
    
    # Create statistics text overlay
    if show_statistics:
        stats_text = f"📊 Label Statistics:\n\n"
        stats_text += f"Sample: {sample_name}\n"
        stats_text += f"Total Labels: {len(labels)}\n"
        stats_text += f"Total Vertices: {total_vertices_highlighted:,}\n"
        stats_text += f"Coverage: {100 * total_vertices_highlighted / len(rm):.2f}%\n\n"
        
        stats_text += f"Individual Labels:\n"
        for stat in region_stats[:10]:  # Show first 10 to avoid clutter
            stats_text += f"  🔸 {stat['label']}: {stat['vertices']:,} vertices "
            stats_text += f"({stat['percentage']:.2f}%)\n"
        
        if len(region_stats) > 10:
            stats_text += f"  ... and {len(region_stats) - 10} more\n"
        
        stats_text += f"\n💡 Controls:\n"
        stats_text += f"  Mouse: Rotate, zoom, pan\n"
        stats_text += f"  R: Reset camera\n"
        stats_text += f"  Q: Quit"
        
        plotter.add_text(
            stats_text, 
            position='upper_left', 
            font_size=9, 
            color='white',
            font='arial'
        )
    
    # Show the plot
    print(f"✓ Launching interactive visualization...")
    plotter.show()
    
    # Print detailed summary
    if show_statistics:
        print(f"\n" + "=" * 60)
        print(f"📋 Detailed Summary:")
        print(f"   Sample file: {sample_name}")
        print(f"   Number of labeled regions: {len(labels)}")
        print(f"   Total vertices highlighted: {total_vertices_highlighted:,}")
        print(f"   Total cortical coverage: {100 * total_vertices_highlighted / len(rm):.2f}%")
        print(f"\n   Label breakdown:")
        for stat in region_stats:
            print(f"      Region {stat['label']:>5}: {stat['vertices']:>6,} vertices ({stat['percentage']:>5.2f}%)")
        print("=" * 60)
    
    return mesh, region_stats

def visualize_multiple_sample_files(sample_filepaths, colors=None, background_color='lightgray'):
    """
    Visualize labels from multiple sample files with different colors
    
    Parameters:
    -----------
    sample_filepaths : list
        List of paths to sample .mat files
    colors : list, optional
        List of colors for each sample file
    background_color : str
        Color for non-labeled regions
    """
    
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
    
    print(f"🧠 Multiple Sample Files Visualizer - {len(sample_filepaths)} Files")
    print("=" * 60)
    
    # Load cortex and mapping data
    print(f"🔧 Loading cortical surface...")
    try:
        cortex = scipy.io.loadmat("../anatomy/fs_cortex_20k.mat")
        region_map = scipy.io.loadmat("../anatomy/fs_cortex_20k_region_mapping.mat")
    except FileNotFoundError:
        print("❌ Error: Could not find the required .mat files")
        return
    
    pos = cortex['pos']
    tri = cortex['tri'] - 1
    rm = region_map['rm'].flatten()
    
    # Create color mapping (0 = background, 1+ = different sample files)
    color_indices = np.zeros(len(rm), dtype=int)
    
    all_stats = []
    
    # Process each sample file
    for file_idx, filepath in enumerate(sample_filepaths):
        print(f"\n📂 Processing file {file_idx + 1}/{len(sample_filepaths)}: {filepath}")
        
        sample_data = load_sample_file(filepath)
        if sample_data is None or 'labels' not in sample_data:
            continue
        
        labels = sample_data['labels'].flatten()
        file_stats = {'filepath': filepath, 'color': colors[file_idx % len(colors)], 'regions': []}
        
        for label in labels:
            label_int = int(label)
            target_mask = rm == label_int
            vertex_count = np.sum(target_mask)
            
            if vertex_count > 0:
                # Assign color index for this sample file
                color_indices[target_mask] = file_idx + 1
                file_stats['regions'].append({'label': label_int, 'vertices': vertex_count})
        
        all_stats.append(file_stats)
    
    # Create mesh
    mesh = pv.PolyData(pos, np.hstack([np.full((tri.shape[0], 1), 3), tri]).astype(np.int32))
    mesh["Sample_Colors"] = color_indices
    
    # Create visualization
    print(f"\n🖥️  Creating visualization...")
    plotter = pv.Plotter(window_size=(1400, 1000))
    
    cmap_colors = [background_color] + [colors[i % len(colors)] for i in range(len(sample_filepaths))]
    
    plotter.add_mesh(
        mesh,
        scalars="Sample_Colors",
        cmap=cmap_colors,
        show_scalar_bar=False
    )
    
    plotter.add_title(f"Multiple Sample Files Comparison\n{len(sample_filepaths)} samples visualized", font_size=14)
    plotter.camera_position = 'iso'
    plotter.show()
    
    return mesh, all_stats

def main():
    """Main function to handle command line or interactive input"""
    
    # Check if file path provided as command line argument
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]
    else:
        # Interactive input
        print("🧠 Sample Labels Visualizer")
        print("=" * 35)
        print("\nAvailable sample files:")
        print("  1. sample_35372.mat")
        print("  2. sample_60988.mat")
        print("  3. sample_78869.mat")
        print("  4. sample_99981.mat")
        
        try:
            choice = input("\nEnter file name or number (1-4): ").strip()
            
            if choice == '1':
                sample_file = "sample_35372.mat"
            elif choice == '2':
                sample_file = "sample_60988.mat"
            elif choice == '3':
                sample_file = "sample_78869.mat"
            elif choice == '4':
                sample_file = "sample_99981.mat"
            else:
                sample_file = choice
                
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            return
    
    # Optional: color argument
    color = 'red'
    if len(sys.argv) > 2:
        color = sys.argv[2]
    
    # Visualize the sample file
    print(f"\n🎯 Visualizing labels from: {sample_file}")
    visualize_sample_labels(sample_file, highlight_color=color)

if __name__ == "__main__":
    # Check for help command
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("""
🧠 Sample Labels Visualizer

Usage:
    python visualize_sample_labels.py <sample_file> [color]
    
Examples:
    python visualize_sample_labels.py sample_35372.mat
    python visualize_sample_labels.py sample_35372.mat blue
    python visualize_sample_labels.py sample_60988.mat orange
    
Interactive Mode:
    python visualize_sample_labels.py
    
Available Sample Files:
    - sample_35372.mat
    - sample_60988.mat
    - sample_78869.mat
    - sample_99981.mat
    
Color Options:
    red, blue, green, orange, purple, yellow, cyan, magenta, etc.
        """)
    else:
        main()
