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
    list_of_files = glob.glob(os.path.join(sim_dir, 'debris_samples_*.csv'))
    if not list_of_files:
        return None, None, None
    filename = max(list_of_files, key=os.path.getctime)
    df = pd.read_csv(filename)
    
    timestamp = filename.split('debris_samples_')[-1].replace('.csv', '')
    report_file_1 = os.path.join(sim_dir, f"analysis_report_{timestamp}.txt")
    report_file_2 = os.path.join(sim_dir, f"analysis_report_{timestamp}.csv.txt")
    report_file = report_file_1 if os.path.exists(report_file_1) else report_file_2
    
    detected_ids = set()
    if os.path.exists(report_file):
        with open(report_file, 'r') as f:
            for line in f:
                if line.startswith("Debris ID "):
                    try:
                        debris_id = int(line.strip().split("Debris ID ")[1].replace(":", ""))
                        detected_ids.add(debris_id)
                    except ValueError:
                        pass
    return df, detected_ids, filename

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
    # ----------------------------------------------------
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
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_octants))
    width = 0.35
    
    ax.bar(x - width/2, octant_counts['Undetected'], width, label='Undetected', color='gray')
    ax.bar(x + width/2, octant_counts['Detected'], width, label='Detected', color='orange')
    
    ax.set_ylabel('Count of Debris Objects')
    ax.set_xlabel('Location Octant (Body X, Body Y, Body Z)')
    ax.set_title('Detection Distribution by Debris Location Octant')
    ax.set_xticks(x)
    ax.set_xticklabels(all_octants)
    ax.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(sim_dir, 'location_octants_histogram.png'), dpi=300)
    
    # ----------------------------------------------------
    # Plot 2: Velocity Direction Map (Azimuth/Elevation)
    # ----------------------------------------------------
    # Calculate Velocity Azimuth and Elevation in Body Frame
    v_norm = np.linalg.norm(v_bf, axis=1)
    deb_df['v_azimuth'] = np.degrees(np.arctan2(v_bf[:, 1], v_bf[:, 0]))  # -180 to 180
    deb_df['v_elevation'] = np.degrees(np.arcsin(v_bf[:, 2] / np.where(v_norm > 0, v_norm, 1e-9))) # -90 to 90
    
    detected_df = deb_df[deb_df['detected']]
    undetected_df = deb_df[~deb_df['detected']]
    
    fig2 = plt.figure(figsize=(10, 6))
    
    # Mollweide projection requires radians, but for a simple scatter, rectangular with degrees is okay
    # We will use purely rectangular map to easily see the clustering
    ax2 = fig2.add_subplot(111)
    
    if len(undetected_df) > 0:
        ax2.scatter(undetected_df['v_azimuth'], undetected_df['v_elevation'],
                   c='gray', alpha=0.5, s=20, label=f'Undetected ({len(undetected_df)})', marker='o')
    if len(detected_df) > 0:
        ax2.scatter(detected_df['v_azimuth'], detected_df['v_elevation'],
                   c='orange', alpha=0.8, s=40, label=f'Detected ({len(detected_df)})', marker='*')
                   
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-90, 90)
    ax2.set_xlabel('Velocity Azimuth in Body Frame (deg)')
    ax2.set_ylabel('Velocity Elevation in Body Frame (deg)')
    ax2.set_title('Velocity Direction Distribution of Debris')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.legend()
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(sim_dir, 'velocity_direction_map.png'), dpi=300)
        
    # ----------------------------------------------------
    # Plot 3: Polar Plot of Velocity Plane Headings
    # ----------------------------------------------------
    num_headings = 72
    # The debris is generated with 72 sequential heading angles
    deb_df['heading_idx'] = deb_df['debris_id'] % num_headings
    deb_df['heading_rad'] = (deb_df['heading_idx'] * 2.0 * np.pi) / num_headings
    
    heading_counts = deb_df.groupby(['heading_rad', 'detected']).size().unstack(fill_value=0)
    heading_counts.rename(columns={False: 'Undetected', True: 'Detected', 0: 'Undetected', 1: 'Detected'}, inplace=True)
    if 'Undetected' not in heading_counts.columns: heading_counts['Undetected'] = 0
    if 'Detected' not in heading_counts.columns: heading_counts['Detected'] = 0
    
    fig3, ax3 = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    angles = heading_counts.index.values
    width_angle = (2.0 * np.pi) / num_headings
    
    # Plot stacked bar chart in polar coordinates
    ax3.bar(angles, heading_counts['Undetected'], width=width_angle, bottom=0.0, 
            color='gray', alpha=0.6, edgecolor='white', label='Undetected')
    ax3.bar(angles, heading_counts['Detected'], width=width_angle, bottom=heading_counts['Undetected'], 
            color='orange', alpha=0.9, edgecolor='white', label='Detected')
    
    ax3.set_title('Detection by Debris Velocity Generation Angle (Circular Velocity Plane)\n', va='bottom')
    ax3.legend(loc='lower left', bbox_to_anchor=(0.9, 0.0))
    
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
