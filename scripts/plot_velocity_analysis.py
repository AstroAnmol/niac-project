import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_latest_sim_dir(results_dir, target_dir=None):
    if target_dir is None:
        list_of_dirs = glob.glob(os.path.join(results_dir, 'Sim_*'))
        if not list_of_dirs:
            return None
        return max(list_of_dirs, key=os.path.getmtime)
    elif not os.path.isabs(target_dir) and not os.path.isdir(target_dir):
        potential_dir = os.path.join(results_dir, target_dir)
        if os.path.isdir(potential_dir):
            return potential_dir
    return target_dir

def load_data(sim_dir):
    # Read the README to find the debris file used for this simulation
    readme_file = os.path.join(sim_dir, "README.txt")
    debris_filename = None

    if os.path.exists(readme_file):
        try:
            with open(readme_file, 'r') as f:
                for line in f:
                    if "Debris file used for detection sim:" in line:
                        # Extract the file path after the colon
                        debris_path = line.split("Debris file used for detection sim:")[-1].strip()
                        debris_filename = debris_path
                        break
        except Exception as e:
            print(f"Warning: Could not read README.txt: {e}")
    
    if debris_filename is None or not os.path.exists(debris_filename):
        print(f"Error: Could not find debris file. Checked: {debris_filename}")
        return None, None, debris_filename
    
    df = pd.read_csv(debris_filename)
    
    # Find the latest debris_oes file in the sim directory
    debris_oes_files = glob.glob(os.path.join(sim_dir, "debris_oes_*.csv"))
    
    detected_ids = set()
    if debris_oes_files:
        debris_oes_file = max(debris_oes_files, key=os.path.getctime)
        print(f"Loading detection info from: {debris_oes_file}")
        oes_df = pd.read_csv(debris_oes_file)
        # Get detected debris IDs
        detected_ids = set(oes_df[oes_df['detected'] == True]['debris_id'].tolist())
    else:
        print(f"Warning: Could not find any debris_oes file in {sim_dir}")
    
    return df, detected_ids, debris_filename

def calculate_body_frame_transform(sat_r, sat_v):
    v_norm = np.linalg.norm(sat_v)
    sat_x_BF = sat_v / v_norm if v_norm > 0 else np.array([1.0, 0.0, 0.0])
    
    r_norm = np.linalg.norm(sat_r)
    sat_z_BF = -sat_r / r_norm if r_norm > 0 else np.array([0.0, 0.0, 1.0])
    
    sat_y_BF = np.cross(sat_z_BF, sat_x_BF)
    y_norm = np.linalg.norm(sat_y_BF)
    if y_norm > 0:
        sat_y_BF /= y_norm
        
    M_BF_to_ECI = np.column_stack((sat_x_BF, sat_y_BF, sat_z_BF))
    M_ECI_to_BF = M_BF_to_ECI.T
    return M_ECI_to_BF

def get_octant(r_bf):
    octant = 0
    if r_bf[0] >= 0: octant += 1
    if r_bf[1] >= 0: octant += 2
    if r_bf[2] >= 0: octant += 4
    return octant

def get_octant_name(r_bf):
    x_sign = "+" if r_bf[0] >= 0 else "-"
    y_sign = "+" if r_bf[1] >= 0 else "-"
    z_sign = "+" if r_bf[2] >= 0 else "-"
    return f"({x_sign}, {y_sign}, {z_sign})"

def analyze_and_plot(sim_dir, save=False):
    df, detected_ids, filename = load_data(sim_dir)
    if df is None:
        print("No data found.")
        return
        
    print(f"Loaded {len(df)-1} debris samples from {filename}")
    
    sat_row = df.iloc[0]
    sat_r = np.array([sat_row['x'], sat_row['y'], sat_row['z']])
    sat_v = np.array([sat_row['vx'], sat_row['vy'], sat_row['vz']])
    
    M_ECI_to_BF = calculate_body_frame_transform(sat_r, sat_v)
    
    deb_df = df.iloc[1:].copy()
    deb_df['debris_id'] = deb_df.index - 1
    deb_df['detected'] = deb_df['debris_id'].isin(detected_ids)
    
    # Calculate relative states in Body Frame
    rel_r = deb_df[['x', 'y', 'z']].values - sat_r
    rel_v = deb_df[['vx', 'vy', 'vz']].values - sat_v
    
    # Transformation
    r_bf = (M_ECI_to_BF @ rel_r.T).T
    v_bf = (M_ECI_to_BF @ rel_v.T).T
    
    deb_df['rx_bf'] = r_bf[:, 0]
    deb_df['ry_bf'] = r_bf[:, 1]
    deb_df['rz_bf'] = r_bf[:, 2]
    
    deb_df['vx_bf'] = v_bf[:, 0]
    deb_df['vy_bf'] = v_bf[:, 1]
    deb_df['vz_bf'] = v_bf[:, 2]
    
    deb_df['octant'] = [get_octant_name(r) for r in r_bf]
    
    # ----------------------------------------------------
    # Plot 1: Histogram of Detected/Undetected by Location Octant (8 Quadrants as referenced by user)
    # Show as percentage of total detected and total undetected
    octant_counts = deb_df.groupby(['octant', 'detected']).size().unstack(fill_value=0)
    octant_counts.rename(columns={False: 'Undetected', True: 'Detected', 0: 'Undetected', 1: 'Detected'}, inplace=True)
    if 'Undetected' not in octant_counts.columns: octant_counts['Undetected'] = 0
    if 'Detected' not in octant_counts.columns: octant_counts['Detected'] = 0
    
    # Ensure all 8 octants are present
    all_octants = ["(+, +, +)", "(+, +, -)", "(+, -, +)", "(+, -, -)",
                   "(-, +, +)", "(-, +, -)", "(-, -, +)", "(-, -, -)"]
    for oct in all_octants:
        if oct not in octant_counts.index:
            octant_counts.loc[oct, 'Undetected'] = 0
            octant_counts.loc[oct, 'Detected'] = 0
    octant_counts = octant_counts.reindex(all_octants)
    
    # Calculate percentages: % of total undetected and % of total detected
    total_undetected = octant_counts['Undetected'].sum()
    total_detected = octant_counts['Detected'].sum()
    
    octant_counts['Undetected_Pct'] = (octant_counts['Undetected'] / total_undetected * 100.0).fillna(0)
    octant_counts['Detected_Pct'] = (octant_counts['Detected'] / total_detected * 100.0).fillna(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_octants))
    width = 0.35
    
    ax.bar(x - width/2, octant_counts['Undetected_Pct'], width, label=f'Undetected (Total: {total_undetected})', color='gray')
    ax.bar(x + width/2, octant_counts['Detected_Pct'], width, label=f'Detected (Total: {total_detected})', color='orange')
    
    ax.set_ylabel('Percentage of Category (%)')
    ax.set_xlabel('Location Octant (Body X, Body Y, Body Z)')
    ax.set_title('Distribution of Detected vs Undetected Debris by Location Octant')
    ax.set_xticks(x)
    ax.set_xticklabels(all_octants)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(sim_dir, 'location_octants_histogram.png'), dpi=300)
    
    # # ----------------------------------------------------
    # # Plot 2: Velocity Direction Map (Azimuth/Elevation)
    # # ----------------------------------------------------
    # # Calculate Velocity Azimuth and Elevation in Body Frame
    # v_norm = np.linalg.norm(v_bf, axis=1)
    # deb_df['v_azimuth'] = np.degrees(np.arctan2(v_bf[:, 1], v_bf[:, 0]))  # -180 to 180
    # deb_df['v_elevation'] = np.degrees(np.arcsin(v_bf[:, 2] / np.where(v_norm > 0, v_norm, 1e-9))) # -90 to 90
    
    # detected_df = deb_df[deb_df['detected']]
    # undetected_df = deb_df[~deb_df['detected']]
    
    # fig2 = plt.figure(figsize=(10, 6))
    
    # # Mollweide projection requires radians, but for a simple scatter, rectangular with degrees is okay
    # # We will use purely rectangular map to easily see the clustering
    # ax2 = fig2.add_subplot(111)
    
    # if len(undetected_df) > 0:
    #     ax2.scatter(undetected_df['v_azimuth'], undetected_df['v_elevation'],
    #                c='gray', alpha=0.5, s=20, label=f'Undetected ({len(undetected_df)})', marker='o')
    # if len(detected_df) > 0:
    #     ax2.scatter(detected_df['v_azimuth'], detected_df['v_elevation'],
    #                c='orange', alpha=0.8, s=40, label=f'Detected ({len(detected_df)})', marker='*')
                   
    # ax2.set_xlim(-180, 180)
    # ax2.set_ylim(-90, 90)
    # ax2.set_xlabel('Velocity Azimuth in Body Frame (deg)')
    # ax2.set_ylabel('Velocity Elevation in Body Frame (deg)')
    # ax2.set_title('Velocity Direction Distribution of Debris')
    # ax2.grid(True, linestyle='--', alpha=0.6)
    # ax2.axhline(0, color='black', linewidth=0.8)
    # ax2.axvline(0, color='black', linewidth=0.8)
    # ax2.legend()
    
    # plt.tight_layout()
    # if save:
    #     plt.savefig(os.path.join(sim_dir, 'velocity_direction_map.png'), dpi=300)
        
    # ----------------------------------------------------
    # Plot 3: Polar Plot of Detection Rate by Velocity Heading
    # (Only for detected positions)
    # ----------------------------------------------------
    num_headings = 72
    # The debris is generated with 72 sequential heading angles
    deb_df['heading_idx'] = deb_df['debris_id'] % num_headings
    deb_df['heading_rad'] = (deb_df['heading_idx'] * 2.0 * np.pi) / num_headings
    
    # Filter to only detected debris
    detected_deb_df = deb_df[deb_df['detected']]
    
    # Count detected debris by heading angle
    heading_counts = detected_deb_df['heading_rad'].value_counts().sort_index()
    
    fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    angles = heading_counts.index.values
    width_angle = (2.0 * np.pi) / num_headings
    counts = heading_counts.values
    
    # Use color gradient based on detection count (actual counts, not normalized)
    max_count = counts.max() if len(counts) > 0 else 1
    colors = plt.cm.Oranges(counts / max_count)
    
    # Plot detected debris count bars with actual counts
    bars = ax3.bar(angles, counts, width=width_angle, bottom=0.0, 
                   color=colors, edgecolor='white', linewidth=1.5)
    
    ax3.set_title(f'Detected Debris Count by Velocity Heading Angle\n(Total Detected: {len(detected_deb_df)})', va='bottom', pad=20)
    ax3.set_ylabel('Number of Detected Debris', labelpad=30)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Oranges, norm=plt.Normalize(vmin=0, vmax=max_count))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax3, pad=0.1)
    cbar.set_label('Count')
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(sim_dir, 'velocity_angle_polar_histogram.png'), dpi=300)

    
    print("Plots generated successfully.")
    if not save:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and plot debris velocity combinations.')
    parser.add_argument('--dir', type=str, help='Path to a Results/Sim_* directory.')
    parser.add_argument('--save', action='store_true', help='Save the plots in the sim directory.')
    args = parser.parse_args()
    
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results')
    sim_dir = get_latest_sim_dir(results_dir, args.dir)
    
    if sim_dir:
        # Default to saving plots since we might run it non-interactively
        analyze_and_plot(sim_dir, save=True)
    else:
        print("No simulation directory found.")
