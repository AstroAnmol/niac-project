import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import argparse

def plot_samples(filename=None):
    if not filename:
        # Get latest CSV in Results directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Results')
        list_of_files = glob.glob(os.path.join(results_dir, 'debris_samples_*.csv'))
        if not list_of_files:
            print(f"No debris sample files found in {results_dir}")
            return
        filename = max(list_of_files, key=os.path.getctime)
    
    print(f"Plotting data from: {filename}")
    
    # Read the data
    df = pd.read_csv(filename)
    
    # The first row is the satellite's state
    sat_x, sat_y, sat_z = df.iloc[0]['x'], df.iloc[0]['y'], df.iloc[0]['z']
    
    # The rest are debris
    deb_df = df.iloc[1:]
    
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # To make everything relative to the satellite (so we can see the empty wake cone easily)
    ax.scatter(deb_df['x'] - sat_x, 
               deb_df['y'] - sat_y, 
               deb_df['z'] - sat_z, 
               c='gray', s=1, alpha=0.5, label='Debris Samples')
               
    ax.scatter([0], [0], [0], c='red', s=50, marker='D', label='Satellite (Origin)')
    
    # Plot formatting
    ax.set_title(f'Debris Samples Relative to Satellite\n({len(deb_df)} points)')
    ax.set_xlabel('Relative X (km)')
    ax.set_ylabel('Relative Y (km)')
    ax.set_zlabel('Relative Z (km)')
    
    # Ensure equal aspect ratio visually via limits
    max_range = max([
        (deb_df['x'] - sat_x).max() - (deb_df['x'] - sat_x).min(),
        (deb_df['y'] - sat_y).max() - (deb_df['y'] - sat_y).min(),
        (deb_df['z'] - sat_z).max() - (deb_df['z'] - sat_z).min()
    ]) / 2.0
    
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize generated debris samples.')
    parser.add_argument('--file', type=str, help='Path to a specific debris CSV file. Defaults to latest in Results/.')
    args = parser.parse_args()
    
    plot_samples(args.file)
