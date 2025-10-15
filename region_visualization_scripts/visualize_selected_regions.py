import pyvista as pv
import scipy.io
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_multiple_regions(region_ids, colors=None, background_color='lightgray', show_labels=True):
    """
    Visualize multiple cortical regions by highlighting them with different colors
    
    Parameters:
    -----------
    region_ids : list or int
        The ID(s) of the region(s) to highlight
    colors : list or str, optional
        Color(s) for the highlighted regions. If None, uses default color cycle
    background_color : str
        Color for all other regions (default: 'lightgray')
    show_labels : bool
        Whether to show region labels and statistics
    """
    
    # Handle single region input
    if isinstance(region_ids, int):
        region_ids = [region_ids]
    
    # Generate default colors if not provided
    if colors is None:
        default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 
                         'lime', 'pink', 'brown', 'gray', 'olive', 'navy', 'teal', 'maroon']
        colors = default_colors[:len(region_ids)]
    elif isinstance(colors, str):
        colors = [colors] * len(region_ids)
    
    # Ensure we have enough colors
    while len(colors) < len(region_ids):
        colors.extend(['red', 'blue', 'green', 'orange', 'purple'])
    
    print(f"🧠 Multiple Regions Visualizer - {len(region_ids)} Region(s)")
    print(f"Region IDs: {region_ids}")
    print("=" * 60)
    
    # Load cortex and mapping data
    print("Loading cortical surface and region mapping...")
    try:
        cortex = scipy.io.loadmat("../anatomy/fs_cortex_20k.mat")
        region_map = scipy.io.loadmat("../anatomy/fs_cortex_20k_region_mapping.mat")
    except FileNotFoundError:
        print("❌ Error: Could not find the required .mat files")
        print("   Make sure the anatomy files are in the correct location")
        return
    
    pos = cortex['pos']  # (20484, 3) - vertex positions
    tri = cortex['tri'] - 1  # Convert to 0-based indexing
    rm = region_map['rm'].flatten()  # (20484,) - region ID per vertex
    
    print(f"✓ Loaded {pos.shape[0]} vertices and {tri.shape[0]} triangles")
    
    # Check if all regions exist
    unique_regions = np.unique(rm)
    missing_regions = [rid for rid in region_ids if rid not in unique_regions]
    if missing_regions:
        print(f"❌ Error: Region ID(s) {missing_regions} not found!")
        print(f"Available region IDs range from {unique_regions.min()} to {unique_regions.max()}")
        print(f"Total regions: {len(unique_regions)}")
        return
    
    # Create mesh
    mesh = pv.PolyData(pos, np.hstack([np.full((tri.shape[0], 1), 3), tri]).astype(np.int32))
    
    # Create color mapping for visualization
    # Initialize with background (0 = background)
    color_indices = np.zeros(len(rm), dtype=int)
    
    region_stats = []
    region_centers = []
    
    # Assign color indices for each region
    for i, region_id in enumerate(region_ids):
        target_mask = rm == region_id
        vertex_count = np.sum(target_mask)
        
        # Assign color index (i+1 because 0 is reserved for background)
        color_indices[target_mask] = i + 1
        
        # Calculate statistics
        if vertex_count > 0:
            region_vertices = pos[target_mask]
            center = np.mean(region_vertices, axis=0)
            extent = np.ptp(region_vertices, axis=0)
            
            region_stats.append({
                'id': region_id,
                'vertices': vertex_count,
                'center': center,
                'extent': extent,
                'color': colors[i],
                'percentage': 100 * vertex_count / len(rm)
            })
            region_centers.append(center)
            
            print(f"✓ Region {region_id}: {vertex_count:,} vertices ({100 * vertex_count / len(rm):.2f}%)")
    
    # Add color mapping to mesh
    mesh["Region_Colors"] = color_indices
    mesh["Region_ID"] = rm
    
    # Create custom colormap
    cmap_colors = [background_color] + colors[:len(region_ids)]
    
    # Create visualization
    print(f"\n🖥️  Creating visualization...")
    plotter = pv.Plotter(window_size=(1200, 900))
    
    # Add mesh with custom coloring
    plotter.add_mesh(
        mesh,
        scalars="Region_Colors",
        cmap=cmap_colors,
        show_scalar_bar=False,
        opacity=1.0
    )
    
    # Create title
    region_list = ", ".join([str(rid) for rid in region_ids])
    total_vertices = sum([stat['vertices'] for stat in region_stats])
    total_percentage = sum([stat['percentage'] for stat in region_stats])
    
    title = f"Regions {region_list} Highlighted\n{len(region_ids)} regions, {total_vertices:,} vertices ({total_percentage:.2f}% of surface)"
    plotter.add_title(title, font_size=14)
    
    # Set camera for good view
    plotter.camera_position = 'iso'
    
    # Add region center points if requested
    if show_labels and region_centers:
        for i, (center, stat) in enumerate(zip(region_centers, region_stats)):
            plotter.add_mesh(pv.Sphere(radius=1.5, center=center), 
                            color=stat['color'], 
                            opacity=0.8,
                            label=f"Region {stat['id']}")
    
    # Create statistics text
    if show_labels:
        stats_text = f"📊 Region Statistics:\n\n"
        for i, stat in enumerate(region_stats):
            stats_text += f"🔸 Region {stat['id']} ({stat['color']}):\n"
            stats_text += f"   Vertices: {stat['vertices']:,}\n"
            stats_text += f"   Coverage: {stat['percentage']:.2f}%\n"
            stats_text += f"   Center: ({stat['center'][0]:.1f}, {stat['center'][1]:.1f}, {stat['center'][2]:.1f})\n\n"
        
        stats_text += f"Controls:\n"
        stats_text += f"- Mouse: Rotate, zoom, pan\n"
        stats_text += f"- R: Reset camera\n"
        stats_text += f"- Q: Quit"
        
        plotter.add_text(stats_text, 
                        position='upper_left', 
                        font_size=9, 
                        color='white')
    
    # Show the plot
    print(f"✓ Launching interactive visualization...")
    plotter.show()
    
    # Print summary statistics
    if show_labels:
        print(f"\n📋 Summary:")
        print(f"   Total regions highlighted: {len(region_ids)}")
        print(f"   Total vertices: {total_vertices:,}")
        print(f"   Total coverage: {total_percentage:.2f}%")
        print(f"   Colors used: {colors[:len(region_ids)]}")
    
    return mesh, region_stats

def visualize_single_region(region_id, highlight_color='red', background_color='lightgray'):
    """
    Wrapper function for backward compatibility - visualize a single region
    """
    return visualize_multiple_regions([region_id], [highlight_color], background_color)

def parse_region_input(input_str):
    """Parse region input string to handle multiple formats"""
    try:
        # Handle comma-separated values
        if ',' in input_str:
            return [int(x.strip()) for x in input_str.split(',')]
        # Handle range notation (e.g., "10-15")
        elif '-' in input_str and input_str.count('-') == 1:
            start, end = input_str.split('-')
            return list(range(int(start.strip()), int(end.strip()) + 1))
        # Handle single value
        else:
            return [int(input_str.strip())]
    except ValueError:
        return None

def main():
    """Main function to handle command line input or interactive input"""
    
    # Check if region ID(s) provided as command line argument
    if len(sys.argv) > 1:
        region_input = sys.argv[1]
        region_ids = parse_region_input(region_input)
        
        if region_ids is None:
            print("❌ Error: Invalid region format")
            print("Usage examples:")
            print("  python visualize_one_region.py 42")
            print("  python visualize_one_region.py 10,15,20")
            print("  python visualize_one_region.py 10-15")
            return
    else:
        # Interactive input
        print("🧠 Multiple Regions Visualizer")
        print("=" * 35)
        print("Enter region ID(s) to visualize:")
        print("  Single: 42")
        print("  Multiple: 10,15,20")
        print("  Range: 10-15")
        
        try:
            region_input = input("\nRegion ID(s): ")
            region_ids = parse_region_input(region_input)
            
            if region_ids is None:
                print("❌ Error: Please enter valid region ID(s)")
                return
                
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            return
    
    # Handle colors
    colors = None
    if len(sys.argv) > 2:
        color_input = sys.argv[2]
        # Parse colors (comma-separated)
        if ',' in color_input:
            colors = [c.strip() for c in color_input.split(',')]
        else:
            colors = [color_input]
    
    # Ask for colors interactively if not provided and multiple regions
    elif len(region_ids) > 1 and len(sys.argv) <= 2:
        try:
            color_choice = input(f"\nUse default colors for {len(region_ids)} regions? (y/n): ").lower()
            if color_choice.startswith('n'):
                color_input = input("Enter colors (comma-separated): ")
                colors = [c.strip() for c in color_input.split(',')]
        except KeyboardInterrupt:
            colors = None  # Use defaults
    
    # Visualize the regions
    print(f"\n🎯 Visualizing {len(region_ids)} region(s): {region_ids}")
    visualize_multiple_regions(region_ids, colors)

def show_available_regions():
    """Helper function to show all available region IDs"""
    print("🔍 Scanning available regions...")
    
    try:
        region_map = scipy.io.loadmat(r"E:\DeepSIF\DeepSif -BrainForge Repo\DeepSIF\anatomy\fs_cortex_20k_region_mapping.mat")
        rm = region_map['rm'].flatten()
        unique_regions = np.unique(rm)
        
        print(f"\n📋 Available Region IDs:")
        print(f"Range: {unique_regions.min()} to {unique_regions.max()}")
        print(f"Total regions: {len(unique_regions)}")
        print(f"\nFirst 20 region IDs: {unique_regions[:20].tolist()}")
        if len(unique_regions) > 20:
            print(f"Last 20 region IDs: {unique_regions[-20:].tolist()}")
        
        return unique_regions
        
    except FileNotFoundError:
        print("❌ Error: Could not find region mapping file")
        return None

if __name__ == "__main__":
    # Check for special commands
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("""
🧠 Multiple Regions Visualizer

Usage:
    python visualize_one_region.py <region_ids> [colors]
    python visualize_one_region.py --list    # Show available regions
    
Region Input Formats:
    Single region:     42
    Multiple regions:  10,15,20,25
    Range of regions:  10-15 (includes 10,11,12,13,14,15)
    
Color Input Formats:
    Single color:      red
    Multiple colors:   red,blue,green,orange
    
Examples:
    python visualize_one_region.py 42                    # Single region in red
    python visualize_one_region.py 42 blue               # Single region in blue
    python visualize_one_region.py 10,15,20              # Multiple regions, default colors
    python visualize_one_region.py 10,15,20 red,blue,green  # Multiple regions, custom colors
    python visualize_one_region.py 10-15                 # Range of regions (10 to 15)
    python visualize_one_region.py 10-15 orange          # Range in orange
    python visualize_one_region.py --list                # Show all available region IDs
    
Interactive Mode:
    python visualize_one_region.py                       # Prompts for input
        """)
        
    elif len(sys.argv) > 1 and sys.argv[1] in ['--list', '-l']:
        show_available_regions()
        
    else:
        main()