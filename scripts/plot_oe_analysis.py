import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse

def plot_oe_analysis(target_dir=None, save=False):
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results')
    
    if target_dir is None:
        list_of_dirs = glob.glob(os.path.join(results_dir, 'Sim_*'))
        if not list_of_dirs:
            print(f"No Sim directories found in {results_dir}")
            return
        target_dir = max(list_of_dirs, key=os.path.getmtime)
    elif not os.path.isabs(target_dir) and not os.path.isdir(target_dir):
        # Allow passing just the folder name like "Sim_20240101_120000"
        potential_dir = os.path.join(results_dir, target_dir)
        if os.path.isdir(potential_dir):
            target_dir = potential_dir
            
    list_of_files = glob.glob(os.path.join(target_dir, 'debris_oes_*.csv'))
    if not list_of_files:
        print(f"No debris OEs files found in {target_dir}")
        return
    file_oe = max(list_of_files, key=os.path.getctime)
        
    print(f"Reading OE data from: {file_oe}")
    
    df_oe = pd.read_csv(file_oe)
    
    # Satellite is the first row
    satellite_oe = df_oe.iloc[0]
    df_oe = df_oe.iloc[1:]
    
    # Try to load corresponding analysis report to find detected IDs
    timestamp = file_oe.split('debris_oes_')[-1].replace('.csv', '')
    report_file_1 = os.path.join(os.path.dirname(file_oe), f"analysis_report_{timestamp}.txt")
    report_file_2 = os.path.join(os.path.dirname(file_oe), f"analysis_report_{timestamp}.csv.txt")
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
    else:
        print(f"Warning: Could not find matching analysis report: {report_file}")
        
    detected_df = df_oe[df_oe['debris_id'].isin(detected_ids)]
    undetected_df = df_oe[~df_oe['debris_id'].isin(detected_ids)]
    
    print(f"Analyzing {len(undetected_df)} undetected vs {len(detected_df)} detected pieces...")
    
    elements = ['a_km', 'e', 'i_deg', 'raan_deg', 'arg_p_deg', 'ta_deg']
    titles = ['Semi-major Axis (a) [km]', 'Eccentricity (e)', 'Inclination (i) [deg]',
              'RAAN [deg]', 'Argument of Periapsis [deg]', 'True Anomaly [deg]']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Orbital Elements Analysis\nDetected ({len(detected_df)}) vs Undetected ({len(undetected_df)})', fontsize=16)
    
    axes = axes.flatten()
    
    for i, (el, title) in enumerate(zip(elements, titles)):
        ax = axes[i]
        
        # Drop infinity for unbounded orbits just in case it breaks histogram scaling
        u_vals = undetected_df[el].replace([float('inf'), float('-inf')], pd.NA).dropna()
        d_vals = detected_df[el].replace([float('inf'), float('-inf')], pd.NA).dropna()
        
        # We use density=True to normalize the histograms so they're visually comparable 
        # despite the huge population size difference between detected/undetected.
        if not u_vals.empty:
            ax.hist(u_vals, bins=30, alpha=0.5, label='Undetected', color='gray', density=False)
        if not d_vals.empty:
            ax.hist(d_vals, bins=30, alpha=0.7, label='Detected', color='orange', density=False)
            
        sat_val = satellite_oe[el]
        if pd.notna(sat_val) and sat_val not in [float('inf'), float('-inf')]:
            ax.axvline(sat_val, color='red', linestyle='--', linewidth=2, label='Satellite OE')
            
        ax.set_title(title)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save:
        save_path = os.path.join(target_dir, 'oe_analysis_plot.png')
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
        
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Orbital Elements of Generated Debris.')
    parser.add_argument('--dir', type=str, help='Path to a Results/Sim_* directory.')
    parser.add_argument('--save', action='store_true', help='Save the plot in the sim directory.')
    args = parser.parse_args()
    
    plot_oe_analysis(args.dir, save=True)
