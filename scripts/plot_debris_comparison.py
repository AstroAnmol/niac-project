import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

def get_sim_info(sim_dir):
    """Extract debris file path and detection info from a sim directory"""
    readme_file = os.path.join(sim_dir, "README.txt")
    debris_filename = None
    
    if os.path.exists(readme_file):
        try:
            with open(readme_file, 'r') as f:
                for line in f:
                    if "Debris file used for detection sim:" in line:
                        debris_path = line.split("Debris file used for detection sim:")[-1].strip()
                        debris_filename = debris_path
                        break
        except Exception as e:
            print(f"Warning: Could not read README from {sim_dir}: {e}")
            return None, None
    
    if debris_filename is None:
        print(f"Error: Could not find debris file reference in {readme_file}")
        return None, None
    
    # Load detection info from debris_oes file
    debris_oes_files = glob.glob(os.path.join(sim_dir, "debris_oes_*.csv"))
    detected_ids = set()
    
    if debris_oes_files:
        debris_oes_file = max(debris_oes_files, key=os.path.getctime)
        try:
            oes_df = pd.read_csv(debris_oes_file)
            detected_ids = set(oes_df[oes_df['detected'] == True]['debris_id'].tolist())
        except Exception as e:
            print(f"Warning: Could not read OE file: {e}")
    
    return debris_filename, detected_ids

def plot_comparison(sim_dirs, save=False):
    """Plot debris detection comparison across multiple simulations"""
    
    if len(sim_dirs) != 4:
        print(f"Error: Expected exactly 4 sim directories, got {len(sim_dirs)}")
        return
    
    print(f"Loading simulation data from 4 directories...")
    
    # Get info from all sims
    sim_data = []
    debris_file = None
    
    for i, sim_dir in enumerate(sim_dirs):
        debris_filename, detected_ids = get_sim_info(sim_dir)
        
        if debris_filename is None:
            print(f"Error: Could not load data from {sim_dir}")
            return
        
        if debris_file is None:
            debris_file = debris_filename
        elif debris_file != debris_filename:
            print(f"Error: Simulations use different debris files!")
            print(f"  Sim 1: {debris_file}")
            print(f"  Sim {i+1}: {debris_filename}")
            return
        
        sim_data.append({
            'dir': sim_dir,
            'detected_ids': detected_ids
        })
        print(f"  Sim {i+1} ({os.path.basename(sim_dir)}): {len(detected_ids)} detections")
    
    print(f"\nAll simulations use the same debris file: {debris_file}")
    
    if not os.path.exists(debris_file):
        print(f"Error: Debris file not found: {debris_file}")
        return
    
    # Load debris samples
    df = pd.read_csv(debris_file)
    
    # The first row is the satellite's state
    sat_x, sat_y, sat_z = df.iloc[0]['x'], df.iloc[0]['y'], df.iloc[0]['z']
    
    # The rest are debris
    deb_df = df.iloc[1:].copy()
    deb_df['debris_id'] = deb_df.index - 1
    
    # Group debris by unique position
    grouped = deb_df.groupby(['x', 'y', 'z'], as_index=False)
    
    unique_positions_data = []
    for (x, y, z), group in grouped:
        # Count how many sims detected this position
        detection_count = 0
        for sim in sim_data:
            if any(group['debris_id'].isin(sim['detected_ids'])):
                detection_count += 1
        
        unique_positions_data.append({
            'x': x,
            'y': y,
            'z': z,
            'vx': group['vx'].iloc[0],
            'vy': group['vy'].iloc[0],
            'vz': group['vz'].iloc[0],
            'detection_count': detection_count
        })
    
    unique_df = pd.DataFrame(unique_positions_data)
    
    # Calculate Satellite Body Frame (BF) rotation matrix
    sat_v = np.array([df.iloc[0]['vx'], df.iloc[0]['vy'], df.iloc[0]['vz']])
    sat_r = np.array([sat_x, sat_y, sat_z])
    
    v_norm = np.linalg.norm(sat_v)
    sat_x_BF = sat_v / v_norm if v_norm > 0 else np.array([1.0, 0.0, 0.0])
    
    r_norm = np.linalg.norm(sat_r)
    sat_z_BF = -sat_r / r_norm if r_norm > 0 else np.array([0.0, 0.0, 1.0])
    
    sat_y_BF = np.cross(sat_z_BF, sat_x_BF)
    y_norm = np.linalg.norm(sat_y_BF)
    if y_norm > 0:
        sat_y_BF /= y_norm
    
    M_BF_to_ECI = np.column_stack((sat_x_BF, sat_y_BF, sat_z_BF))
    
    # Transform to body frame
    rel_pos = unique_df[['x', 'y', 'z']].values - sat_r
    pos_BF = rel_pos @ M_BF_to_ECI
    
    x_bf = pos_BF[:, 0]
    y_bf = pos_BF[:, 1]
    z_bf = pos_BF[:, 2]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color map: 0=red, 1=orange, 2=yellow, 3=light green, 4=dark green
    colors_map = {0: 'red', 1: 'orange', 2: 'yellow', 3: 'lightgreen', 4: 'darkgreen'}
    
    # Plot debris grouped by detection count (skip red - positions with 0 detections)
    for count in range(1, 5):  # Skip count=0 (red)
        mask = unique_df['detection_count'] == count
        if mask.any():
            ax.scatter(x_bf[mask], y_bf[mask], z_bf[mask], 
                      c=colors_map[count], s=20, alpha=0.7, 
                      label=f'Detected in {count}/4 sims ({mask.sum()} positions)')
    
    # Plot Satellite Origin
    ax.scatter([0], [0], [0], c='black', s=100, marker='D', label='Satellite (Origin)', zorder=10)
    
    # Plot formatting
    ax.set_title(f'Debris Detection Comparison Across 4 Simulations\n({len(unique_df)} unique positions)')
    ax.set_xlabel('Body X (Velocity dir, km)')
    ax.set_ylabel('Body Y (Cross-track, km)')
    ax.set_zlabel('Body Z (Radial dir, Nadir, km)')
    
    # Set equal aspect ratio
    max_range = max([
        x_bf.max() - x_bf.min(),
        y_bf.max() - y_bf.min(),
        z_bf.max() - z_bf.min()
    ]) / 2.0 if len(unique_df) > 0 else 1.0
    
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    # Plot wake cone
    L = max_range * 1.5
    tan20 = np.tan(np.deg2rad(60))
    
    apex = np.array([0, 0, 0])
    c1 = np.array([-L,  L*tan20,  L*tan20])
    c2 = np.array([-L, -L*tan20,  L*tan20])
    c3 = np.array([-L, -L*tan20, -L*tan20])
    c4 = np.array([-L,  L*tan20, -L*tan20])
    
    verts = [
        [apex, c1, c2],
        [apex, c2, c3],
        [apex, c3, c4],
        [apex, c4, c1]
    ]
    
    cone = Poly3DCollection(verts, alpha=0.1, facecolor='cyan', edgecolor='blue', linewidths=0.5)
    wake_patch = mpatches.Patch(color='cyan', alpha=0.3, label='Wake Cone')
    
    ax.add_collection3d(cone)
    
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    
    if save:
        # Save to a new directory for comparison plots
        comp_dir = "Results/Comparison_Plots"
        os.makedirs(comp_dir, exist_ok=True)
        
        # Create a name based on the sim folders
        sim_names = "_".join([os.path.basename(d).split("_")[1] for d in sim_dirs])
        save_path = os.path.join(comp_dir, f'debris_comparison_{sim_names}.png')
        plt.savefig(save_path, dpi=300)
        print(f"\nPlot saved to: {save_path}")
    
    plt.tight_layout()
    # plt.show()
    
    # Print statistics
    print(f"\n=== Detection Statistics ===")
    for count in range(5):
        num_positions = (unique_df['detection_count'] == count).sum()
        print(f"Detected in {count}/4 sims: {num_positions} positions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare debris detection across 4 simulations.')
    parser.add_argument('sim_dirs', nargs=4, help='4 paths to Sim_* directories')
    parser.add_argument('--save', action='store_true', help='Save the plot to Results/Comparison_Plots/')
    args = parser.parse_args()
    
    plot_comparison(args.sim_dirs, save=args.save)
