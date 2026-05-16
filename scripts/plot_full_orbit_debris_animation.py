import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import argparse
import re

def parse_readme_for_true_anomaly(readme_path):
    """Extract True Anomaly from README.txt"""
    try:
        with open(readme_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'a_km, e, i_deg, omega_deg, Omega_deg, theta_deg' in line:
                    if i + 1 < len(lines):
                        oe_line = lines[i + 1].strip()
                        values = [float(x.strip()) for x in oe_line.split(',')]
                        if len(values) >= 6:
                            return values[5]
    except Exception as e:
        pass
    return None

def calculate_body_frame_transform(sat_r, sat_v):
    """Calculate transformation matrix from ECI to satellite body frame"""
    v_norm = np.linalg.norm(sat_v)
    sat_x_BF = sat_v / v_norm if v_norm > 0 else np.array([1.0, 0.0, 0.0])
    
    r_norm = np.linalg.norm(sat_r)
    sat_z_BF = -sat_r / r_norm if r_norm > 0 else np.array([0.0, 0.0, 1.0])
    
    sat_y_BF = np.cross(sat_z_BF, sat_x_BF)
    y_norm = np.linalg.norm(sat_y_BF)
    if y_norm > 0:
        sat_y_BF /= y_norm
    
    M_BF_to_ECI = np.column_stack((sat_x_BF, sat_y_BF, sat_z_BF))
    return M_BF_to_ECI

def load_sim_debris_data(sim_dir, output_txt_path=None):
    """Load debris data for a single simulation"""
    readme_path = os.path.join(sim_dir, "README.txt")
    detection_files = glob.glob(os.path.join(sim_dir, "detection_results_*.csv"))
    
    if not detection_files or not os.path.exists(readme_path):
        return None
    
    true_anomaly = parse_readme_for_true_anomaly(readme_path)
    if true_anomaly is None:
        return None
    
    # Read debris file path and satellite state from README
    debris_file = None
    sat_state = None
    boom_angles = []
    debris_samples = None
    search_radius = None
    sim_time = None
    
    try:
        with open(readme_path, 'r') as f:
            content = f.read()
            # Extract debris file
            if "Debris file used for detection sim:" in content:
                debris_file = content.split("Debris file used for detection sim:")[-1].split('\n')[0].strip()
            # Extract satellite state - look for Initial Cartesian State
            if "Initial Cartesian State" in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "Initial Cartesian State" in line:
                        if i + 1 < len(lines):
                            state_line = lines[i + 1].strip()
                            try:
                                sat_state = [float(x) for x in state_line.split() if x]
                            except:
                                pass
                        break
    except Exception as e:
        # print(f"[DEBUG] Error reading {readme_path}: {e}")
        return None
    
    # Read output.txt for search parameters and boom angles
    if output_txt_path and os.path.exists(output_txt_path):
        try:
            with open(output_txt_path, 'r') as f:
                content = f.read()
                # Extract boom angles
                if "Boom angles:" in content:
                    boom_line = content.split("Boom angles:")[-1].split('\n')[0].strip()
                    try:
                        # Try splitting by comma first, then by whitespace
                        if ',' in boom_line:
                            boom_angles = [float(x.strip()) for x in boom_line.split(',') if x.strip()]
                        else:
                            boom_angles = [float(x.strip()) for x in boom_line.split() if x.strip()]
                        # print(f"[DEBUG] Extracted boom_angles: {boom_angles}")
                    except Exception as e:
                        # print(f"[DEBUG] Failed to parse boom angles from '{boom_line}': {e}")
                        boom_angles = []
                # else:
                #     print(f"[DEBUG] 'Boom angles:' not found in output.txt")
                # Extract search parameters
                if "Debris parameters:" in content or "debris parameters" in content.lower():
                    for line in content.split('\n'):
                        if "num_samples" in line.lower() or "debris samples" in line.lower():
                            try:
                                debris_samples = int(''.join(filter(str.isdigit, line.split('=')[-1] if '=' in line else line.split(':')[-1])))
                                # print(f"[DEBUG] Extracted debris_samples: {debris_samples}")
                            except Exception as e:
                                pass
                                # print(f"[DEBUG] Failed to parse debris_samples: {e}")
                        if "search_radius" in line.lower() or "detection radius" in line.lower():
                            try:
                                part = line.split('=')[-1] if '=' in line else line.split(':')[-1]
                                search_radius = float(''.join(c for c in part if c.isdigit() or c == '.'))
                                # print(f"[DEBUG] Extracted search_radius: {search_radius}")
                            except Exception as e:
                                pass
                                # print(f"[DEBUG] Failed to parse search_radius: {e}")
                        if "final_time" in line.lower() or "simulation time" in line.lower():
                            try:
                                part = line.split('=')[-1] if '=' in line else line.split(':')[-1]
                                sim_time = float(''.join(c for c in part if c.isdigit() or c == '.'))
                                # print(f"[DEBUG] Extracted sim_time: {sim_time}")
                            except Exception as e:
                                pass
                                # print(f"[DEBUG] Failed to parse sim_time: {e}")
        except Exception as e:
            pass
            # print(f"[DEBUG] Error reading {output_txt_path}: {e}")
    
    # Set fallback defaults if not found in output.txt
    if debris_samples is None:
        debris_samples = 10000
    if search_radius is None:
        search_radius = 10.0
    if sim_time is None:
        sim_time = 2.0
    
    if not debris_file:
        # print(f"[DEBUG] No debris file found in {os.path.basename(sim_dir)}")
        return None
    
    # Resolve debris file path
    results_full_orbit_path = os.path.dirname(sim_dir)
    potential_paths = [
        debris_file,
        os.path.join(results_full_orbit_path, debris_file),
        os.path.join(os.path.dirname(results_full_orbit_path), debris_file),
    ]
    
    debris_df = None
    resolved_path = None
    for potential_path in potential_paths:
        if os.path.exists(potential_path):
            try:
                debris_df = pd.read_csv(potential_path)
                resolved_path = potential_path
                break
            except:
                pass
    
    if debris_df is None:
        # print(f"[DEBUG] Could not load debris file for {os.path.basename(sim_dir)}")
        # print(f"        Tried paths: {potential_paths}")
        return None
    
    # Load detection results
    detection_file = max(detection_files, key=os.path.getctime)
    try:
        detection_df = pd.read_csv(detection_file)
        detected_ids = set(detection_df['debris_id'].values)
    except:
        detected_ids = set()
    
    # Extract satellite state
    if sat_state is None or len(sat_state) < 6:
        sat_x, sat_y, sat_z = debris_df.iloc[0]['x'], debris_df.iloc[0]['y'], debris_df.iloc[0]['z']
        sat_vx, sat_vy, sat_vz = debris_df.iloc[0]['vx'], debris_df.iloc[0]['vy'], debris_df.iloc[0]['vz']
    else:
        sat_x, sat_y, sat_z, sat_vx, sat_vy, sat_vz = sat_state[:6]
    
    # Get debris positions
    deb_df = debris_df.iloc[1:].copy()
    deb_df['debris_id'] = deb_df.index - 1
    deb_df['detected'] = deb_df['debris_id'].isin(detected_ids)
    
    # Group by unique position
    grouped = deb_df.groupby(['x', 'y', 'z'], as_index=False)
    unique_positions_data = []
    
    for (x, y, z), group in grouped:
        is_detected = group['detected'].any()
        unique_positions_data.append({
            'x': x,
            'y': y,
            'z': z,
            'vx': group['vx'].iloc[0],
            'vy': group['vy'].iloc[0],
            'vz': group['vz'].iloc[0],
            'detected': is_detected,
            'count': len(group)
        })
    
    unique_df = pd.DataFrame(unique_positions_data)
    
    # Transform to body frame
    sat_r = np.array([sat_x, sat_y, sat_z])
    sat_v = np.array([sat_vx, sat_vy, sat_vz])
    M_BF = calculate_body_frame_transform(sat_r, sat_v)
    
    # Transform all debris positions
    rel_pos = unique_df[['x', 'y', 'z']].values - sat_r
    pos_BF = rel_pos @ M_BF
    
    unique_df['x_bf'] = pos_BF[:, 0]
    unique_df['y_bf'] = pos_BF[:, 1]
    unique_df['z_bf'] = pos_BF[:, 2]
    
    return {
        'true_anomaly': true_anomaly,
        'sim_dir': sim_dir,
        'debris_df': unique_df,
        'detected_count': unique_df['detected'].sum(),
        'total_count': len(unique_df),
        'detection_rate': unique_df['detected'].sum() / len(unique_df) if len(unique_df) > 0 else 0,
        'boom_angles': boom_angles,
        'debris_samples': debris_samples,
        'search_radius': search_radius,
        'sim_time': sim_time
    }

def load_all_sim_debris_data(results_dir):
    """Load debris data from all simulations"""
    sim_dirs = sorted(glob.glob(os.path.join(results_dir, 'Sim_*')))
    
    # Find output.txt in results directory
    output_txt_path = os.path.join(results_dir, 'output.txt')
    if not os.path.exists(output_txt_path):
        # Try parent directory
        output_txt_path = os.path.join(os.path.dirname(results_dir), 'output.txt')
    
    if not os.path.exists(output_txt_path):
        # print(f"[WARNING] output.txt not found. Trying {output_txt_path}")
        output_txt_path = None
    # else:
        # print(f"[INFO] Using output.txt: {output_txt_path}")
    
    sim_data_list = []
    for sim_dir in sim_dirs:
        sim_data = load_sim_debris_data(sim_dir, output_txt_path)
        if sim_data is not None:
            sim_data_list.append(sim_data)
    
    # Sort by true anomaly
    sim_data_list.sort(key=lambda x: x['true_anomaly'])
    
    return sim_data_list

def create_debris_animation(sim_data_list, save=False, output_dir=None):
    """Create animation of debris cloud as satellite progresses through orbit"""
    
    if not sim_data_list:
        print("Error: No simulation data available")
        return
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 7))
    
    # 3D plot for debris cloud
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 2D plot for statistics
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    
    # Pre-compute common limits
    all_x_bf = []
    all_y_bf = []
    all_z_bf = []
    
    for sim_data in sim_data_list:
        df = sim_data['debris_df']
        all_x_bf.extend(df['x_bf'].values)
        all_y_bf.extend(df['y_bf'].values)
        all_z_bf.extend(df['z_bf'].values)
    
    all_x_bf = np.array(all_x_bf)
    all_y_bf = np.array(all_y_bf)
    all_z_bf = np.array(all_z_bf)
    
    x_lim = (np.min(all_x_bf) * 1.1, np.max(all_x_bf) * 1.1)
    y_lim = (np.min(all_y_bf) * 1.1, np.max(all_y_bf) * 1.1)
    z_lim = (np.min(all_z_bf) * 1.1, np.max(all_z_bf) * 1.1)
    
    # Initialize scatter plots
    scatter_undetected = None
    scatter_detected = None
    scatter_origin = None
    text_info = ax2.text(0.05, 0.95, '', fontsize=11, verticalalignment='top',
                         family='monospace', transform=ax2.transAxes)
    
    # Calculate overall statistics
    total_detected_all = sum(sim['detected_count'] for sim in sim_data_list)
    total_debris_all = sum(sim['total_count'] for sim in sim_data_list)
    mean_detected = total_detected_all / len(sim_data_list) if sim_data_list else 0
    overall_detection_rate = total_detected_all / total_debris_all if total_debris_all > 0 else 0
    
    def animate(frame):
        nonlocal scatter_undetected, scatter_detected, scatter_origin
        
        # Clear previous plots
        ax1.clear()
        
        sim_data = sim_data_list[frame]
        debris_df = sim_data['debris_df']
        
        # Separate detected and undetected
        detected = debris_df[debris_df['detected']]
        undetected = debris_df[~debris_df['detected']]
        
        # Plot undetected
        if len(undetected) > 0:
            ax1.scatter(undetected['x_bf'], undetected['y_bf'], undetected['z_bf'],
                       c='gray', s=3, alpha=0.4, label=f'Undetected ({len(undetected)})')
        
        # Plot detected
        if len(detected) > 0:
            ax1.scatter(detected['x_bf'], detected['y_bf'], detected['z_bf'],
                       c='orange', s=15, alpha=1.0, label=f'Detected ({len(detected)})')
        
        # Plot satellite origin
        ax1.scatter([0], [0], [0], c='red', s=100, marker='D', label='Satellite')
        
        # Set labels and limits
        ax1.set_xlabel('Body X (Velocity, km)', fontsize=10)
        ax1.set_ylabel('Body Y (Cross-track, km)', fontsize=10)
        ax1.set_zlabel('Body Z (Nadir, km)', fontsize=10)
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        ax1.set_zlim(z_lim)
        ax1.legend(fontsize=9, loc='upper right')
        ax1.set_title(f'Debris Cloud in Satellite Body Frame\nTrue Anomaly: {sim_data["true_anomaly"]:.1f}°',
                     fontsize=12, fontweight='bold')
        
        # Update statistics text
        stats_text = f"""
FULL ORBIT DEBRIS ANIMATION
{'='*45}

Position: {frame + 1}/{len(sim_data_list)}
True Anomaly:        {sim_data['true_anomaly']:.1f}°

Detection Statistics:
  Total Debris:      {sim_data['total_count']}
  Detected:          {sim_data['detected_count']}
  Detection Rate:    {sim_data['detection_rate']:.1%}

Debris Distribution:
  Undetected:        {len(undetected)}
  Detected:          {len(detected)}

Overall Statistics:
  Mean Detected:     {mean_detected:.1f}
  Overall Rate:      {overall_detection_rate:.1%}

Sensor Configuration:
  Boom Angles:"""
        if len(sim_data['boom_angles']) > 0:
            angles_str = ', '.join([f"{a:.1f}°" for a in sim_data['boom_angles']])
            stats_text += f"    {angles_str}\n"
        else:
            stats_text += "    N/A\n"
        
        stats_text += f"""
Search Parameters:
  Debris Samples:    {sim_data['debris_samples']:,}
  Search Radius:     {sim_data['search_radius']:.1f} km
  Sim Time:          {sim_data['sim_time']:.1f} s
        """
        text_info.set_text(stats_text)
        
        return ax1, text_info
    
    anim = FuncAnimation(fig, animate, frames=len(sim_data_list),
                        interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    
    if save and output_dir:
        output_path = os.path.join(output_dir, 'full_orbit_debris_animation.gif')
        print(f"Saving animation to {output_path}... (this may take a few minutes)")
        writer = PillowWriter(fps=10)
        anim.save(output_path, writer=writer)
        print(f"Saved: {output_path}")
    
    plt.show()
    return fig, anim

def main():
    parser = argparse.ArgumentParser(description='Create debris cloud animation for full orbit')
    parser.add_argument('--results-dir', type=str,
                       default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           'Results_full_orbit', 'Results'),
                       help='Path to Results_full_orbit/Results directory')
    parser.add_argument('--save', action='store_true', help='Save animation to file')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    print(f"Loading debris data from: {args.results_dir}")
    sim_data_list = load_all_sim_debris_data(args.results_dir)
    
    if not sim_data_list:
        print("Error: No simulation data found!")
        return
    
    print(f"Loaded {len(sim_data_list)} simulations")
    
    output_dir = os.path.dirname(args.results_dir) if args.save else None
    create_debris_animation(sim_data_list, save=args.save, output_dir=output_dir)

if __name__ == '__main__':
    main()
