import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse

def plot_samples(target_dir=None, save=False):
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results')
    
    if target_dir is None:
        list_of_dirs = glob.glob(os.path.join(results_dir, 'Sim_*'))
        if not list_of_dirs:
            print(f"No Sim directories found in {results_dir}")
            return
        target_dir = max(list_of_dirs, key=os.path.getmtime)
    elif not os.path.isabs(target_dir) and not os.path.isdir(target_dir):
        potential_dir = os.path.join(results_dir, target_dir)
        if os.path.isdir(potential_dir):
            target_dir = potential_dir
            
    list_of_files = glob.glob(os.path.join(target_dir, 'debris_oes_*.csv'))
    if not list_of_files:
        print(f"No debris orbital elements files found in {target_dir}")
        return
    
    filename = max(list_of_files, key=os.path.getctime)
    
    # Read the README to find the debris file used for this simulation
    readme_file = os.path.join(target_dir, "README.txt")
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
    
    if debris_filename is None:
        print(f"Warning: Could not find debris file reference in {readme_file}")
        return

    print(f"Plotting data from: {debris_filename} and {filename}")

    # Read the orbital elements data
    oe_df = pd.read_csv(filename)
    
    # Extract detected debris IDs from the 'detected' column
    detected_ids = set(oe_df[oe_df['detected'] == True]['debris_id'].tolist())
    
    # Now read the corresponding debris_samples file for actual positions
    # debris_samples_filename = filename.replace("debris_oes_", "debris_samples_")
    if not os.path.exists(debris_filename):
        print(f"Error: Could not find corresponding debris samples file: {debris_filename}")
        return
    
    df = pd.read_csv(debris_filename)

    # The first row is the satellite's state
    sat_x, sat_y, sat_z = df.iloc[0]['x'], df.iloc[0]['y'], df.iloc[0]['z']
    
    # The rest are debris
    deb_df = df.iloc[1:].copy()
    deb_df['debris_id'] = deb_df.index - 1
    
    # Group debris by unique position (72 debris at same position have different velocities)
    # For each unique position, mark as detected if ANY of the 72 debris at that position is detected
    grouped = deb_df.groupby(['x', 'y', 'z'], as_index=False)
    
    unique_positions_data = []
    for (x, y, z), group in grouped:
        # Check if any debris at this position is detected
        is_detected = any(group['debris_id'].isin(detected_ids))
        unique_positions_data.append({
            'x': x,
            'y': y,
            'z': z,
            'vx': group['vx'].iloc[0],
            'vy': group['vy'].iloc[0],
            'vz': group['vz'].iloc[0],
            'is_detected': is_detected
        })
    
    unique_df = pd.DataFrame(unique_positions_data)
    detected_df = unique_df[unique_df['is_detected']]
    undetected_df = unique_df[~unique_df['is_detected']]
    
    # Calculate Satellite Body Frame (BF) rotation matrix
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.patches as mpatches
    
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
    
    # Transformation function from relative ECI to BF
    def to_bf(sub_df):
        if sub_df.empty:
            return np.array([]), np.array([]), np.array([])
        rel_pos = sub_df[['x', 'y', 'z']].values - sat_r
        pos_BF = rel_pos @ M_BF_to_ECI
        return pos_BF[:, 0], pos_BF[:, 1], pos_BF[:, 2]
    
    ux_bf, uy_bf, uz_bf = to_bf(undetected_df)
    dx_bf, dy_bf, dz_bf = to_bf(detected_df)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot debris in Body Frame
    if len(ux_bf) > 0:
        ax.scatter(ux_bf, uy_bf, uz_bf, 
                   c='gray', s=1, alpha=0.5, label=f'Undetected Positions ({len(undetected_df)})')
               
    if len(dx_bf) > 0:
        ax.scatter(dx_bf, dy_bf, dz_bf, 
                   c='orange', s=10, alpha=1.0, label=f'Detected Positions ({len(detected_df)})')
               
    # Plot Satellite Origin
    ax.scatter([0], [0], [0], c='red', s=50, marker='D', label='Satellite (Origin)')
    
    # Plot formatting
    ax.set_title(f'Unique Debris Positions in Satellite Body Frame\n({len(unique_df)} unique positions)')
    ax.set_xlabel('Body X (Velocity dir, km)')
    ax.set_ylabel('Body Y (Cross-track, km)')
    ax.set_zlabel('Body Z (Radial dir, Nadir, km)')
    
    # Ensure equal aspect ratio visually via limits based on BF coordinates
    all_x = np.concatenate([ux_bf, dx_bf]) if len(unique_df) > 0 else np.array([0])
    all_y = np.concatenate([uy_bf, dy_bf]) if len(unique_df) > 0 else np.array([0])
    all_z = np.concatenate([uz_bf, dz_bf]) if len(unique_df) > 0 else np.array([0])
    
    max_range = max([
        all_x.max() - all_x.min(),
        all_y.max() - all_y.min(),
        all_z.max() - all_z.min()
    ]) / 2.0 if len(unique_df) > 0 else 1.0
    
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    # Plot wake planes (pyramid) expanding backwards from satellite in Body Frame
    L = max_range * 1.5  # Length of wake cone
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
    
    cone = Poly3DCollection(verts, alpha=0.1, facecolor='cyan', edgecolor='blue', linewidths=0.5, label='Wake Cone')
    wake_patch = mpatches.Patch(color='cyan', alpha=0.3, label='Wake Cone')
    
    ax.add_collection3d(cone)
    
    # Plot satellite booms and sensors in Body Frame
    sat_x_size, sat_y_size, sat_z_size = 3.0 * 0.0001, 2.0 * 0.0001, 2.0 * 0.0001
    boom_length = 0.001
    
    alpha_x = 54.7356 * np.pi / 180
    alpha_z = 45.0 * np.pi / 180
    
    sv0 = np.array([np.cos(alpha_x), np.sin(alpha_z)*np.sin(alpha_x), np.cos(alpha_z)*np.sin(alpha_x)])
    sv1 = np.array([np.cos(alpha_x), -np.sin(alpha_z)*np.sin(alpha_x), np.cos(alpha_z)*np.sin(alpha_x)])
    sv2 = np.array([np.cos(alpha_x), -np.sin(alpha_z)*np.sin(alpha_x), -np.cos(alpha_z)*np.sin(alpha_x)])
    sv3 = np.array([np.cos(alpha_x), np.sin(alpha_z)*np.sin(alpha_x), -np.cos(alpha_z)*np.sin(alpha_x)])
    
    corner_1_BF = np.array([sat_x_size/2, sat_y_size/2, sat_z_size/2])
    corner_2_BF = np.array([sat_x_size/2, -sat_y_size/2, sat_z_size/2])
    corner_3_BF = np.array([sat_x_size/2, -sat_y_size/2, -sat_z_size/2])
    corner_4_BF = np.array([sat_x_size/2, sat_y_size/2, -sat_z_size/2])
    
    sen_1_BF = corner_1_BF + boom_length * sv0
    sen_2_BF = corner_2_BF + boom_length * sv1
    sen_3_BF = corner_3_BF + boom_length * sv2
    sen_4_BF = corner_4_BF + boom_length * sv3
    
    ax.plot([corner_1_BF[0], sen_1_BF[0]], [corner_1_BF[1], sen_1_BF[1]], [corner_1_BF[2], sen_1_BF[2]], color='black', linewidth=1.5, zorder=5)
    ax.plot([corner_2_BF[0], sen_2_BF[0]], [corner_2_BF[1], sen_2_BF[1]], [corner_2_BF[2], sen_2_BF[2]], color='black', linewidth=1.5, zorder=5)
    ax.plot([corner_3_BF[0], sen_3_BF[0]], [corner_3_BF[1], sen_3_BF[1]], [corner_3_BF[2], sen_3_BF[2]], color='black', linewidth=1.5, zorder=5)
    ax.plot([corner_4_BF[0], sen_4_BF[0]], [corner_4_BF[1], sen_4_BF[1]], [corner_4_BF[2], sen_4_BF[2]], color='black', linewidth=1.5, zorder=5)
    
    ax.scatter([sen_1_BF[0], sen_2_BF[0], sen_3_BF[0], sen_4_BF[0]],
               [sen_1_BF[1], sen_2_BF[1], sen_3_BF[1], sen_4_BF[1]],
               [sen_1_BF[2], sen_2_BF[2], sen_3_BF[2], sen_4_BF[2]],
               color='magenta', s=10, zorder=6, label='Sensors')
    
    handles, labels = ax.get_legend_handles_labels()
    handles.append(wake_patch)
    labels.append("Wake Cone")
    ax.legend(handles=handles, labels=labels)
    plt.tight_layout()
    
    if save:
        save_path = os.path.join(os.path.dirname(filename), 'debris_plot.png')
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
        
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize generated debris samples.')
    parser.add_argument('--dir', type=str, help='Path to a Results/Sim_* directory.')
    parser.add_argument('--save', action='store_true', help='Save the plot in the sim directory.')
    args = parser.parse_args()
    
    plot_samples(args.dir, save=True)
