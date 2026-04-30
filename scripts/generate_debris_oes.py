import os
import glob
import csv
import math
import sys

def rv_to_oe(r, v):
    mu = 398600.4418
    r_mag = math.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    v_mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    
    h = [
        r[1]*v[2] - r[2]*v[1],
        r[2]*v[0] - r[0]*v[2],
        r[0]*v[1] - r[1]*v[0]
    ]
    h_mag = math.sqrt(h[0]**2 + h[1]**2 + h[2]**2)
    n = [-h[1], h[0], 0]
    n_mag = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    
    r_dot_v = r[0]*v[0] + r[1]*v[1] + r[2]*v[2]
    e_vec = [
        ((v_mag**2 - mu/r_mag)*r[0] - r_dot_v*v[0]) / mu,
        ((v_mag**2 - mu/r_mag)*r[1] - r_dot_v*v[1]) / mu,
        ((v_mag**2 - mu/r_mag)*r[2] - r_dot_v*v[2]) / mu
    ]
    e_mag = math.sqrt(e_vec[0]**2 + e_vec[1]**2 + e_vec[2]**2)
    
    epsilon = (v_mag**2)/2 - mu/r_mag
    if abs(epsilon) < 1e-12:
        a = float('inf')
    else:
        a = -mu / (2 * epsilon)
        
    inc = math.acos(max(-1.0, min(1.0, h[2] / h_mag))) if h_mag > 0 else 0
    
    if n_mag == 0:
        raan = 0
    else:
        raan = math.acos(max(-1.0, min(1.0, n[0] / n_mag)))
        if n[1] < 0:
            raan = 2*math.pi - raan
            
    if n_mag == 0 or e_mag == 0:
        arg_p = 0
    else:
        n_dot_e = n[0]*e_vec[0] + n[1]*e_vec[1] + n[2]*e_vec[2]
        arg_p = math.acos(max(-1.0, min(1.0, n_dot_e / (n_mag * e_mag))))
        if e_vec[2] < 0:
            arg_p = 2*math.pi - arg_p
            
    if e_mag == 0:
        ta = 0
    else:
        e_dot_r = e_vec[0]*r[0] + e_vec[1]*r[1] + e_vec[2]*r[2]
        ta = math.acos(max(-1.0, min(1.0, e_dot_r / (e_mag * r_mag))))
        if r_dot_v < 0:
            ta = 2*math.pi - ta
            
    return {
        'a': a, 'e': e_mag, 'i': math.degrees(inc),
        'raan': math.degrees(raan), 'arg_p': math.degrees(arg_p), 'ta': math.degrees(ta)
    }

def process_samples(target_dir=None):
    results_dir = "./Results/"

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

    # Find the latest detection_results file in the target directory
    search_pattern = os.path.join(target_dir, "detection_results_*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        print("No detection results found in", target_dir)
        return
        
    latest_results_file = max(files, key=os.path.getmtime)
    print(f"Analyzing: {os.path.basename(latest_results_file)}")

    timestamp_part = latest_results_file.split('detection_results_')[-1]
    
    # Read the README to find the debris file used for this simulation
    readme_file = os.path.join(target_dir, "README.txt")
    matching_samples_file = None
    
    if os.path.exists(readme_file):
        try:
            with open(readme_file, 'r') as f:
                for line in f:
                    if "Debris file used for detection sim:" in line:
                        # Extract the file path after the colon
                        debris_path = line.split("Debris file used for detection sim:")[-1].strip()
                        matching_samples_file = debris_path
                        break
        except Exception as e:
            print(f"Warning: Could not read README.txt: {e}")
    
    if matching_samples_file is None:
        print(f"Warning: Could not find debris file reference in {readme_file}")
        return
    
    report_file = os.path.join(target_dir, f"analysis_report_{timestamp_part}.txt")

    output_filename = os.path.join(target_dir, f"debris_oes_{timestamp_part}")

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

    print(f"Reading: {debris_path}")
    print(f"Generating: {output_filename}")
    
    with open(debris_path, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(['debris_id', 'a_km', 'e', 'i_deg', 'raan_deg', 'arg_p_deg', 'ta_deg', 'detected'])
        
        try:
            # Skip header
            next(reader)
        except StopIteration:
            print("File is empty or lacks required headers.")
            return
            
        count = 0
        for i, row in enumerate(reader):
            if len(row) >= 6:
                r = [float(row[0]), float(row[1]), float(row[2])]
                v = [float(row[3]), float(row[4]), float(row[5])]
                
                oe = rv_to_oe(r, v)
                
                actual_id = -1 if i == 0 else i - 1
                is_detected = actual_id in detected_ids
                
                writer.writerow([
                    actual_id, 
                    f"{oe['a']:.6f}", 
                    f"{oe['e']:.6f}", 
                    f"{oe['i']:.6f}", 
                    f"{oe['raan']:.6f}", 
                    f"{oe['arg_p']:.6f}", 
                    f"{oe['ta']:.6f}",
                    is_detected
                ])
                count += 1
                
    print(f"Successfully wrote {count} orbital elements to {os.path.basename(output_filename)}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate OEs from Debris Samples.')
    parser.add_argument('target_dir', nargs='?', type=str, help='Path to a Results/Sim_* directory.')
    args = parser.parse_args()
    
    process_samples(args.target_dir)
