import pyvista as pv
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path

print("🧠 Enhanced Cortical Region Visualization")
print("=" * 50)

# Load cortex and mapping data
print("Loading cortical surface and region mapping...")
cortex = scipy.io.loadmat("../anatomy/fs_cortex_20k.mat")
region_map = scipy.io.loadmat("../anatomy/fs_cortex_20k_region_mapping.mat")

pos = cortex['pos']  # (20484, 3) - vertex positions
tri = cortex['tri'] - 1  # Convert to 0-based indexing for Python
rm = region_map['rm'].flatten()  # (20484,) - region ID per vertex

print(f"✓ Loaded {pos.shape[0]} vertices and {tri.shape[0]} triangles")
print(f"✓ Found {len(np.unique(rm))} unique regions")

# Create FreeSurfer region labels (simplified version)
# In practice, you'd load this from a proper FreeSurfer parcellation file
def create_region_labels():
    """Create region ID to anatomical name mapping"""
    # This is a simplified mapping - in reality you'd load from FreeSurfer's LUT
    region_names = {}
    unique_regions = np.unique(rm)
    
    # Common FreeSurfer regions (partial list for demonstration)
    fs_labels = {
        0: "Unknown/Medial Wall",
        1: "Left Frontal Pole",
        2: "Left Superior Frontal",
        3: "Left Rostral Middle Frontal",
        4: "Left Caudal Middle Frontal",
        5: "Left Pars Opercularis",
        6: "Left Pars Triangularis",
        7: "Left Pars Orbitalis",
        8: "Left Lateral Orbitofrontal",
        9: "Left Medial Orbitofrontal",
        10: "Left Precentral",
        11: "Left Paracentral",
        12: "Left Superior Parietal",
        13: "Left Inferior Parietal",
        14: "Left Supramarginal",
        15: "Left Angular",
        16: "Left Precuneus",
        17: "Left Superior Temporal",
        18: "Left Middle Temporal",
        19: "Left Inferior Temporal",
        20: "Left Banks STS",
        # Add more regions as needed...
    }
    
    # Fill in remaining regions with generic names
    for region_id in unique_regions:
        if region_id in fs_labels:
            region_names[region_id] = fs_labels[region_id]
        else:
            # Determine hemisphere and create generic label
            if region_id < 500:  # Arbitrary split for demonstration
                hemisphere = "Left"
            else:
                hemisphere = "Right"
            region_names[region_id] = f"{hemisphere} Region {region_id}"
    
    return region_names

region_labels = create_region_labels()

# Create detailed region statistics
def analyze_regions():
    """Analyze region properties"""
    unique_regions = np.unique(rm)
    region_stats = []
    
    for region_id in unique_regions:
        mask = rm == region_id
        vertex_count = np.sum(mask)
        region_vertices = pos[mask]
        
        if vertex_count > 0:
            # Calculate region properties
            center = np.mean(region_vertices, axis=0)
            extent = np.ptp(region_vertices, axis=0)  # Range in each dimension
            surface_area = vertex_count * 0.1  # Approximate (would need proper calculation)
            
            region_stats.append({
                'Region_ID': int(region_id),
                'Name': region_labels.get(region_id, f"Region {region_id}"),
                'Vertex_Count': vertex_count,
                'Center_X': center[0],
                'Center_Y': center[1], 
                'Center_Z': center[2],
                'Extent_X': extent[0],
                'Extent_Y': extent[1],
                'Extent_Z': extent[2],
                'Approx_Area_mm2': surface_area
            })
    
    return pd.DataFrame(region_stats).sort_values('Region_ID')

print("\nAnalyzing region properties...")
region_df = analyze_regions()
print(f"✓ Analyzed {len(region_df)} regions")

# Display region summary
print(f"\n📊 Region Summary:")
print(f"Total regions: {len(region_df)}")
print(f"Total vertices: {len(rm)}")
print(f"Average vertices per region: {len(rm) / len(region_df):.1f}")

# Show first few regions
print(f"\n🏷️  First 10 Regions:")
print(region_df[['Region_ID', 'Name', 'Vertex_Count', 'Approx_Area_mm2']].head(10).to_string(index=False))

# Create enhanced mesh with multiple scalar fields
print(f"\nCreating enhanced 3D mesh...")
mesh = pv.PolyData(pos, np.hstack([np.full((tri.shape[0], 1), 3), tri]).astype(np.int32))

# Add multiple scalar fields
mesh["Region_ID"] = rm
mesh["Vertex_Index"] = np.arange(len(pos))

# Create region size field (vertices per region)
region_sizes = region_df.set_index('Region_ID')['Vertex_Count'].to_dict()
mesh["Region_Size"] = np.array([region_sizes.get(rid, 0) for rid in rm])

# Create hemisphere field
hemisphere = np.where(pos[:, 0] < 0, 0, 1)  # 0=Left, 1=Right based on X coordinate
mesh["Hemisphere"] = hemisphere

# Interactive visualization
print(f"\n🖥️  Launching interactive visualization...")

class InteractiveBrainViewer:
    def __init__(self, mesh, region_df):
        self.mesh = mesh
        self.region_df = region_df
        self.plotter = pv.Plotter(window_size=(1200, 800))
        self.current_scalar = "Region_ID"
        
    def setup_plot(self):
        """Setup the initial plot"""
        # Add main mesh
        self.plotter.add_mesh(
            self.mesh, 
            scalars=self.current_scalar, 
            cmap="tab20",  # Better for discrete regions
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Region ID',
                'label_font_size': 10,
                'title_font_size': 12,
                'n_labels': 10
            }
        )
        
        # Add title and camera
        self.plotter.add_title("Enhanced Cortical Regions Visualization\nClick on regions to see details", font_size=14)
        self.plotter.camera_position = 'iso'
        
        # Add text widget for region info
        self.info_widget = self.plotter.add_text("Click on a region to see details", 
                                                 position='upper_left', 
                                                 font_size=10, 
                                                 color='white')
        
        # Enable point picking
        self.plotter.enable_point_picking(callback=self.point_callback, 
                                         show_point=True, 
                                         color='red',
                                         point_size=10)
        
        # Add controls text
        controls_text = """
        Controls:
        - Click: Select region and show info
        - Mouse: Rotate, zoom, pan
        - R: Reset camera
        - Q: Quit
        """
        self.plotter.add_text(controls_text, position='lower_left', font_size=8, color='lightgray')
        
    def point_callback(self, point):
        """Callback when user clicks on a point"""
        # Find closest vertex
        distances = np.linalg.norm(self.mesh.points - point, axis=1)
        closest_vertex = np.argmin(distances)
        
        # Get region info
        region_id = self.mesh["Region_ID"][closest_vertex]
        region_info = self.region_df[self.region_df['Region_ID'] == region_id]
        
        if not region_info.empty:
            info = region_info.iloc[0]
            info_text = f"""
            Selected Region Information:
            
            🏷️  Region ID: {info['Region_ID']}
            📍 Name: {info['Name']}
            🔢 Vertices: {info['Vertex_Count']:,}
            📏 Approx Area: {info['Approx_Area_mm2']:.1f} mm²
            📊 Center: ({info['Center_X']:.1f}, {info['Center_Y']:.1f}, {info['Center_Z']:.1f})
            📐 Extent: ({info['Extent_X']:.1f}, {info['Extent_Y']:.1f}, {info['Extent_Z']:.1f})
            """
            
            # Update info widget
            self.plotter.textActor.SetInput(info_text)
            print(f"\n🎯 Selected: {info['Name']} (ID: {info['Region_ID']})")
            print(f"   Vertices: {info['Vertex_Count']:,}, Area: {info['Approx_Area_mm2']:.1f} mm²")
    
    def show(self):
        """Display the interactive plot"""
        self.setup_plot()
        self.plotter.show()

# Launch interactive viewer
viewer = InteractiveBrainViewer(mesh, region_df)
viewer.show()

# Save region information to CSV
output_file = "cortical_regions_analysis.csv"
region_df.to_csv(output_file, index=False)
print(f"\n💾 Region analysis saved to: {output_file}")

print(f"\n✅ Visualization complete!")
print(f"📋 Region mapping details:")
print(f"   - Total regions mapped: {len(region_df)}")
print(f"   - Interactive selection enabled")
print(f"   - Region details saved to CSV")
print(f"   - Click on any region to see its anatomical information!")
