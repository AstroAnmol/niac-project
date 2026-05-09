import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import argparse
import re

def parse_readme_for_true_anomaly(readme_path):
    """Extract True Anomaly from README.txt"""
    try:
        with open(readme_path, 'r') as f:
            lines = f.readlines()
            # Find the OE line with the values
            for i, line in enumerate(lines):
                if 'a_km, e, i_deg, omega_deg, Omega_deg, theta_deg' in line:
                    # Next line should have the values
                    if i + 1 < len(lines):
                        oe_line = lines[i + 1].strip()
                        values = [float(x.strip()) for x in oe_line.split(',')]
                        if len(values) >= 6:
                            return values[5]  # theta_deg is the 6th value (0-indexed: 5)
    except Exception as e:
        print(f"Warning: Could not parse README from {readme_path}: {e}")
    return None


def load_sim_data(sim_dir):
    """Load simulation data from a single sim directory"""
    readme_path = os.path.join(sim_dir, "README.txt")
    detection_files = glob.glob(os.path.join(sim_dir, "detection_results_*.csv"))
    
    if not detection_files or not os.path.exists(readme_path):
        return None
    
    true_anomaly = parse_readme_for_true_anomaly(readme_path)
    if true_anomaly is None:
        return None
    
    # Read debris file path from README to get total debris count
    debris_file = None
    try:
        with open(readme_path, 'r') as f:
            for line in f:
                if "Debris file used for detection sim:" in line:
                    debris_file = line.split("Debris file used for detection sim:")[-1].strip()
                    break
    except Exception as e:
        print(f"Warning: Could not read README from {readme_path}: {e}")
    
    # Get total debris count from debris samples file
    total_debris = 0
    if debris_file:
        # The path in README is relative to Results_full_orbit, so we need to resolve it
        # Results_full_orbit/Results/Sim_*/ contains the sim dir
        # So we go up to Results_full_orbit/Results/ and append the debris path
        results_full_orbit_path = os.path.dirname(sim_dir)  # Gets Results_full_orbit/Results
        
        # Try multiple path resolutions
        potential_paths = [
            debris_file,  # absolute path
            os.path.join(results_full_orbit_path, debris_file),  # relative to Results_full_orbit/Results
            os.path.join(os.path.dirname(results_full_orbit_path), debris_file),  # relative to Results_full_orbit
        ]
        
        for potential_path in potential_paths:
            if os.path.exists(potential_path):
                try:
                    debris_df = pd.read_csv(potential_path)
                    total_debris = len(debris_df) - 1  # Subtract 1 for satellite row
                    break
                except Exception as e:
                    pass
    
    if total_debris == 0:
        return None
    
    detection_file = max(detection_files, key=os.path.getctime)
    try:
        detection_df = pd.read_csv(detection_file)
        detected_count = len(detection_df)  # All rows are detected
        
        # Create detection array (for heatmap - mark detected ones as 1)
        detection_array = detection_df['detected'].values
        debris_ids = detection_df['debris_id'].values
        
        return {
            'true_anomaly': true_anomaly,
            'detected_count': detected_count,
            'total_debris': total_debris,
            'detection_rate': detected_count / total_debris if total_debris > 0 else 0,
            'detection_array': detection_array,
            'debris_ids': debris_ids,
            'detection_df': detection_df,
            'sim_dir': sim_dir
        }
    except Exception as e:
        print(f"Warning: Could not load detection data from {detection_file}: {e}")
        return None

def load_all_sim_data(results_dir):
    """Load data from all Sim_* directories"""
    sim_dirs = sorted(glob.glob(os.path.join(results_dir, 'Sim_*')))
    
    sim_data_list = []
    for sim_dir in sim_dirs:
        sim_data = load_sim_data(sim_dir)
        if sim_data is not None:
            sim_data_list.append(sim_data)
    
    # Sort by true anomaly
    sim_data_list.sort(key=lambda x: x['true_anomaly'])
    
    return sim_data_list

def create_heatmap_data(sim_data_list):
    """Create a heatmap where rows are debris and columns are true anomaly angles"""
    # Collect all unique debris IDs across all simulations
    all_debris_ids = set()
    for sim_data in sim_data_list:
        all_debris_ids.update(sim_data['debris_ids'])
    
    all_debris_ids = sorted(list(all_debris_ids))
    
    # Create heatmap matrix: rows = debris, columns = true anomaly
    heatmap_data = np.zeros((len(all_debris_ids), len(sim_data_list)))
    true_anomalies = []
    
    for col_idx, sim_data in enumerate(sim_data_list):
        true_anomalies.append(sim_data['true_anomaly'])
        # Create set of detected debris IDs for this simulation
        detected_ids = set(sim_data['debris_ids'])
        
        for row_idx, debris_id in enumerate(all_debris_ids):
            heatmap_data[row_idx, col_idx] = 1 if debris_id in detected_ids else 0
    
    return heatmap_data, all_debris_ids, true_anomalies

def plot_detection_rate_and_heatmap(sim_data_list, save=False, output_dir=None):
    """Plot detection rate vs true anomaly and heatmap"""
    
    true_anomalies = [d['true_anomaly'] for d in sim_data_list]
    detection_rates = [d['detection_rate'] for d in sim_data_list]
    detected_counts = [d['detected_count'] for d in sim_data_list]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Detection Rate vs True Anomaly
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(true_anomalies, detection_rates, 'b-o', linewidth=2, markersize=6, label='Detection Rate')
    ax1.fill_between(true_anomalies, detection_rates, alpha=0.3)
    ax1.set_xlabel('True Anomaly (degrees)', fontsize=12)
    ax1.set_ylabel('Detection Rate', fontsize=12)
    ax1.set_title('Debris Detection Rate Across Full Orbit', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 365)
    ax1.set_ylim(0, max(detection_rates) * 1.1)
    ax1.legend(fontsize=10)
    
    # Add count labels on secondary axis
    ax1_secondary = ax1.twinx()
    ax1_secondary.plot(true_anomalies, detected_counts, 'r--s', linewidth=2, markersize=5, 
                       label='Detected Count', alpha=0.7)
    ax1_secondary.set_ylabel('Number of Detected Debris', fontsize=12, color='r')
    ax1_secondary.tick_params(axis='y', labelcolor='r')
    
    # Plot 2: Heatmap
    ax2 = plt.subplot(2, 1, 2)
    heatmap_data, debris_ids, true_anom = create_heatmap_data(sim_data_list)
    
    # Only show a subset of debris for readability (every Nth debris)
    step = max(1, len(debris_ids) // 100)  # Show max 100 debris rows
    heatmap_display = heatmap_data[::step, :]
    debris_display = [debris_ids[i] for i in range(0, len(debris_ids), step)]
    
    im = ax2.imshow(heatmap_display, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax2.set_xlabel('True Anomaly (degrees)', fontsize=12)
    ax2.set_ylabel('Debris ID (subset)', fontsize=12)
    ax2.set_title('Detection Pattern Across Orbit (Red=Not Detected, Green=Detected)', 
                  fontsize=14, fontweight='bold')
    
    # Set x-axis to show true anomalies
    ax2.set_xticks(np.arange(0, len(true_anom), max(1, len(true_anom)//10)))
    ax2.set_xticklabels([f'{true_anom[i]:.0f}°' for i in ax2.get_xticks()], rotation=45)
    
    # Set y-axis to show sample debris IDs
    ax2.set_yticks(np.arange(0, len(debris_display), max(1, len(debris_display)//10)))
    ax2.set_yticklabels([f'{int(debris_display[i])}' for i in ax2.get_yticks()])
    
    cbar = plt.colorbar(im, ax=ax2, label='Detected (1) / Not Detected (0)')
    
    plt.tight_layout()
    
    if save and output_dir:
        output_path = os.path.join(output_dir, 'full_orbit_detection_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.show()
    return fig, heatmap_data, debris_ids, true_anomalies

def create_orbit_animation(sim_data_list, save=False, output_dir=None):
    """Create animation of satellite progressing around orbit with detection visualization"""
    
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._xyz = (x, y, z)
            self._dxdydz = (dx, dy, dz)

        def draw(self, renderer):
            x1, y1, z1 = self._xyz
            dx, dy, dz = self._dxdydz
            
            xs = np.linspace(x1, x1 + dx, 2)
            ys = np.linspace(y1, y1 + dy, 2)
            zs = np.linspace(z1, z1 + dz, 2)
            
            xs_proj, ys_proj = proj_transform(xs, ys, zs, self.axes.M)
            self.set_positions((xs_proj[0], xs_proj[1]), (ys_proj[0], ys_proj[1]))
            super().draw(renderer)
    
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Orbital position indicator
    ax1 = fig.add_subplot(121)
    true_anomalies = [d['true_anomaly'] for d in sim_data_list]
    detection_rates = [d['detection_rate'] for d in sim_data_list]
    
    # Plot full orbit curve
    ax1.plot(true_anomalies, detection_rates, 'b-', linewidth=2, alpha=0.5)
    ax1.fill_between(true_anomalies, detection_rates, alpha=0.2)
    
    # Initialize line and scatter plot for animation
    line_anim, = ax1.plot([], [], 'ro', markersize=12, label='Current Position')
    
    ax1.set_xlabel('True Anomaly (degrees)', fontsize=11)
    ax1.set_ylabel('Detection Rate', fontsize=11)
    ax1.set_title('Detection Rate vs True Anomaly', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 365)
    ax1.set_ylim(0, max(detection_rates) * 1.1)
    ax1.legend(fontsize=10)
    
    # Plot 2: Statistics text
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    text_anim = ax2.text(0.1, 0.9, '', fontsize=12, verticalalignment='top', 
                         family='monospace', transform=ax2.transAxes)
    
    def animate(frame):
        if frame >= len(sim_data_list):
            frame = frame % len(sim_data_list)
        
        sim_data = sim_data_list[frame]
        ta = sim_data['true_anomaly']
        
        # Update position marker
        line_anim.set_data([ta], [sim_data['detection_rate']])
        
        # Update statistics text
        stats_text = f"""
ORBITAL POSITION ANALYSIS
{'='*40}

True Anomaly:        {ta:.1f}°
Detection Rate:      {sim_data['detection_rate']:.1%}
Detected Debris:     {sim_data['detected_count']}/{sim_data['total_debris']}

Position around orbit:  {frame+1}/72

Orbital Parameters:
  a = 7121 km
  e = 0.063
  i = 135°
  
Debris Cloud:
  Samples: 10,000
  Search Radius: 10 km
  Simulation Time: 2 s
        """
        text_anim.set_text(stats_text)
        
        return line_anim, text_anim
    
    anim = FuncAnimation(fig, animate, frames=len(sim_data_list), 
                        interval=100, blit=True, repeat=True)
    
    if save and output_dir:
        output_path = os.path.join(output_dir, 'orbit_animation.gif')
        writer = PillowWriter(fps=10)
        anim.save(output_path, writer=writer)
        print(f"Saved animation: {output_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, anim

def print_orbit_statistics(sim_data_list):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("FULL ORBIT DETECTION ANALYSIS SUMMARY")
    print("="*60)
    
    true_anomalies = [d['true_anomaly'] for d in sim_data_list]
    detection_rates = [d['detection_rate'] for d in sim_data_list]
    detected_counts = [d['detected_count'] for d in sim_data_list]
    
    print(f"Total Simulations:          {len(sim_data_list)}")
    print(f"True Anomaly Range:         {min(true_anomalies):.1f}° to {max(true_anomalies):.1f}°")
    print(f"Increment:                  5° (uniform)")
    print(f"\nDetection Statistics:")
    print(f"  Average Detection Rate:   {np.mean(detection_rates):.1%}")
    print(f"  Max Detection Rate:       {np.max(detection_rates):.1%} at TA={true_anomalies[np.argmax(detection_rates)]:.1f}°")
    print(f"  Min Detection Rate:       {np.min(detection_rates):.1%} at TA={true_anomalies[np.argmin(detection_rates)]:.1f}°")
    print(f"  Std Deviation:            {np.std(detection_rates):.1%}")
    print(f"\nDetection Counts:")
    print(f"  Average:                  {np.mean(detected_counts):.0f} debris")
    print(f"  Max:                      {np.max(detected_counts):.0f} debris")
    print(f"  Min:                      {np.min(detected_counts):.0f} debris")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze full orbit detection patterns')
    parser.add_argument('--results-dir', type=str, 
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                           'Results_full_orbit', 'Results'),
                       help='Path to Results_full_orbit/Results directory')
    parser.add_argument('--save', action='store_true', help='Save plots to files')
    parser.add_argument('--animation', action='store_true', help='Generate animation')
    parser.add_argument('--stats-only', action='store_true', help='Print statistics only')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    print(f"Loading simulation data from: {args.results_dir}")
    sim_data_list = load_all_sim_data(args.results_dir)
    
    if not sim_data_list:
        print("Error: No simulation data found!")
        return
    
    print(f"Loaded {len(sim_data_list)} simulations")
    
    # Print statistics
    print_orbit_statistics(sim_data_list)
    
    if args.stats_only:
        return
    
    output_dir = os.path.dirname(args.results_dir) if args.save else None
    
    # Create main plot
    print("\nGenerating detection rate and heatmap plots...")
    plot_detection_rate_and_heatmap(sim_data_list, save=args.save, output_dir=output_dir)
    
    # Create animation
    if args.animation:
        print("Generating orbit animation...")
        create_orbit_animation(sim_data_list, save=args.save, output_dir=output_dir)

if __name__ == '__main__':
    main()
