import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
        results_full_orbit_path = os.path.dirname(sim_dir)  # Results_full_orbit/Results
        # Strip leading path component (e.g. "Results/") to resolve inside results_dir
        debris_file_stripped = os.path.join(*debris_file.replace('\\', '/').split('/')[1:]) \
            if '/' in debris_file or '\\' in debris_file else debris_file
        potential_paths = [
            debris_file,
            os.path.join(results_full_orbit_path, debris_file),
            os.path.join(os.path.dirname(results_full_orbit_path), debris_file),
            os.path.join(results_full_orbit_path, debris_file_stripped),
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

def load_j2_propagation(results_dir):
    """Load J2 propagation data (time, RAAN, AoP, true anomaly) from sat_propagation_j2_file.csv.
    Returns a DataFrame, or None if the file is not found."""
    j2_path = os.path.join(results_dir, 'sat_propagation_j2_file.csv')
    if not os.path.exists(j2_path):
        j2_path = os.path.join(os.path.dirname(results_dir), 'sat_propagation_j2_file.csv')
    if not os.path.exists(j2_path):
        return None
    try:
        df = pd.read_csv(j2_path)
        # Rename columns to short identifiers for convenience
        df = df.rename(columns={
            'Time (sec)': 'time',
            'RAAN (deg)': 'raan',
            'Argument of Periapsis (deg)': 'aop',
            'True Anomaly (deg)': 'ta',
        })
        return df[['time', 'raan', 'aop', 'ta']]
    except Exception as e:
        print(f'Warning: Could not load J2 propagation file: {e}')
        return None


def load_all_sim_data(results_dir):
    """Load data from all Sim_* directories, attaching J2-propagated time per sim."""
    sim_dirs = sorted(glob.glob(os.path.join(results_dir, 'Sim_*')))
    j2_df = load_j2_propagation(results_dir)

    # Collect valid sims in chronological (directory-sorted) order
    sim_data_list = []
    for sim_dir in sim_dirs:
        sim_data = load_sim_data(sim_dir)
        if sim_data is not None:
            sim_data_list.append(sim_data)

    # Each sim corresponds 1-to-1 with a J2 row by chronological index
    if j2_df is not None:
        for i, sim_data in enumerate(sim_data_list):
            if i < len(j2_df):
                row = j2_df.iloc[i]
                sim_data['time_sec'] = float(row['time'])
                sim_data['raan']     = float(row['raan'])
                sim_data['aop']      = float(row['aop'])
            else:
                sim_data['time_sec'] = None
                sim_data['raan']     = None
                sim_data['aop']      = None
    else:
        for sim_data in sim_data_list:
            sim_data['time_sec'] = None
            sim_data['raan']     = None
            sim_data['aop']      = None

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

def _add_cyclic_xaxis(ax, times, values, row_label, period=360.0,
                       tick_step=60.0, y_offset=-0.12, color='#333333'):
    """Draw a row of degree-ticks beneath the x-axis at every `tick_step` degrees.

    Returns a list of (x_norm, x_raw) tuples for each tick placed, so callers
    can pass these positions to _add_corresponding_xaxis for other angle rows.
    """
    times  = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    ax_xmin, ax_xmax = ax.get_xlim()
    span = ax_xmax - ax_xmin

    # Row label on the far left
    ax.text(-0.01, y_offset, f'{row_label}:',
            transform=ax.transAxes, fontsize=8, color=color,
            ha='right', va='center', fontweight='bold', clip_on=False)

    # Unwrap so we can interpolate through 0/360 wrap-arounds
    values_unwrapped = np.unwrap(np.deg2rad(values)) * (180.0 / np.pi)

    v_min = values_unwrapped.min()
    v_max = values_unwrapped.max()
    first_multiple = np.ceil(v_min / tick_step) * tick_step
    targets = np.arange(first_multiple, v_max + tick_step * 0.01, tick_step)

    tick_positions = []  # (x_norm, x_raw)
    for target in targets:
        diffs = values_unwrapped - target
        sign_changes = np.where(np.diff(np.sign(diffs)))[0]
        if len(sign_changes) == 0:
            continue
        idx = sign_changes[0]
        x0, x1 = times[idx], times[idx + 1]
        v0, v1 = diffs[idx], diffs[idx + 1]
        x_cross = x0 - v0 * (x1 - x0) / (v1 - v0)

        x_norm = (x_cross - ax_xmin) / span
        if not (0.0 <= x_norm <= 1.0):
            continue

        label_deg = target % period
        ax.text(x_norm, y_offset, f'{label_deg:.0f}°',
                transform=ax.transAxes, fontsize=7.5, color=color,
                ha='center', va='center', clip_on=False)
        ax.plot([x_norm, x_norm], [y_offset + 0.015, y_offset - 0.015],
                transform=ax.transAxes, color=color, linewidth=0.8,
                clip_on=False)
        tick_positions.append((x_norm, x_cross))

    return tick_positions


def _add_corresponding_xaxis(ax, tick_positions, times, values, row_label,
                              y_offset=-0.12, color='#333333', fmt='{:.1f}°'):
    """At each x-position in `tick_positions` (from _add_cyclic_xaxis), label
    the interpolated value of a different angle series (`values` vs `times`).
    """
    times  = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)

    ax.text(-0.01, y_offset, f'{row_label}:',
            transform=ax.transAxes, fontsize=8, color=color,
            ha='right', va='center', fontweight='bold', clip_on=False)

    for x_norm, x_raw in tick_positions:
        val = float(np.interp(x_raw, times, values))
        ax.text(x_norm, y_offset, fmt.format(val),
                transform=ax.transAxes, fontsize=7.5, color=color,
                ha='center', va='center', clip_on=False)
        ax.plot([x_norm, x_norm], [y_offset + 0.015, y_offset - 0.015],
                transform=ax.transAxes, color=color, linewidth=0.8,
                clip_on=False)


def plot_detection_rate_and_heatmap(sim_data_list, save=False, output_dir=None):
    """Plot detection rate vs elapsed time (with J2 propagation).
    The x-axis shows time in seconds; beneath the standard tick labels three
    additional rows show the cyclic True Anomaly, RAAN and AoP with ticks at
    every 60 degrees.
    """

    has_time = sim_data_list[0]['time_sec'] is not None

    if has_time:
        x_vals  = np.array([d['time_sec'] for d in sim_data_list])
        x_label = 'Elapsed Time (s)'
        x_title = 'Elapsed Time (s) — J2 propagation'
    else:
        x_vals  = np.array([d['true_anomaly'] for d in sim_data_list])
        x_label = 'True Anomaly (°)'
        x_title = 'True Anomaly (°)'

    true_anomalies  = np.array([d['true_anomaly'] for d in sim_data_list])
    raan_vals       = np.array([d['raan'] if d['raan'] is not None else 0.0
                                for d in sim_data_list])
    aop_vals        = np.array([d['aop']  if d['aop']  is not None else 0.0
                                for d in sim_data_list])
    detection_rates = np.array([d['detection_rate'] for d in sim_data_list])
    detected_counts = np.array([d['detected_count'] for d in sim_data_list])

    # ------------------------------------------------------------------ figure
    # bottom=0.26 gives room for 3 annotation rows beneath the single plot
    fig = plt.figure(figsize=(16, 7))
    fig.subplots_adjust(left=0.09, right=0.94, top=0.92, bottom=0.26)

    # --------------------------------------------------------- rate & count plot
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(x_vals, detection_rates, color='steelblue', linewidth=2,
             marker='o', markersize=5, label='Detection Rate')
    ax1.fill_between(x_vals, detection_rates, alpha=0.18, color='steelblue')
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Detection Rate', fontsize=11, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_title(f'Debris Detection Rate vs {x_title}',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.25)
    margin = (x_vals[-1] - x_vals[0]) * 0.02
    ax1.set_xlim(x_vals[0] - margin, x_vals[-1] + margin)
    ax1.set_ylim(0, max(detection_rates) * 1.15)

    ax1r = ax1.twinx()
    ax1r.plot(x_vals, detected_counts, color='tomato', linewidth=1.8,
              linestyle='--', marker='s', markersize=4,
              label='Detected Count', alpha=0.8)
    ax1r.set_ylabel('Detected Count', fontsize=11, color='tomato')
    ax1r.tick_params(axis='y', labelcolor='tomato')

    # Combined legend
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper right')

    # Cyclic annotation rows below the plot.
    # TA: ticks at every 60°; RAAN & AoP: corresponding values at the same positions.
    if has_time:
        ta_ticks = _add_cyclic_xaxis(ax1, x_vals, true_anomalies, 'TA',
                                     tick_step=60.0, y_offset=-0.18, color='#1a5276')
        _add_corresponding_xaxis(ax1, ta_ticks, x_vals, raan_vals, 'RAAN',
                                 y_offset=-0.30, color='#145a32')
        _add_corresponding_xaxis(ax1, ta_ticks, x_vals, aop_vals, 'AoP',
                                 y_offset=-0.42, color='#6e2f1a')

    if save and output_dir:
        output_path = os.path.join(output_dir, 'full_orbit_detection_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_path}')

    plt.show()
    return fig, x_vals.tolist()

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
